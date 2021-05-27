#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import os
import math
import json

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src, src_str =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src, batch.src_str
        query_id = batch.query_id
        '''
        try:
            query_id = batch.query_id
        except:
            query_id = None
        '''
        translations = []
        for b in range(batch_size):
            if self.args.dataset == 'qg_ranking_test':
                if self.args.encoder == "bert":
                    pred_sents = [" ".join(self.vocab.convert_ids_to_tokens([int(n) for n in each])).replace(" ##", "") for each in preds[b]]
                elif self.args.encoder == "roberta":
                    pred_sents = [self.vocab.decode([int(n) for n in each]).replace("<s>", "").replace("</s>", "") for each in preds[b]]
            elif self.args.encoder == 'roberta':
                pred_sents = self.vocab.decode([int(n) for n in preds[b][0]]).replace("<s>", "").replace("</s>", "")
            elif self.args.encoder == 'bert':
                pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
                pred_sents = ' '.join(pred_sents).replace(' ##','')
            elif self.args.encoder == 'zh_bert' and self.args.dataset == 'paraphrase':
                pred_sents = [self.vocab.convert_ids_to_tokens([int(n) for n in pred]) for pred in preds[b]]
                pred_sents = [''.join(pred).replace(' ##', '') for pred in pred_sents]
            elif self.args.encoder == 'zh_bert':
                pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
                pred_sents = ''.join(pred_sents).replace('##','')
            gold_sent = tgt_str[b]
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])

            if self.args.encoder == 'roberta':
                raw_src = self.vocab.decode([int(t) for t in src[b]])
            else:
                raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
                raw_src = ' '.join(raw_src)
            if self.args.dataset == 'faq': 
                translation = (pred_sents, gold_sent, src_str[b], query_id[b], pred_score[b])
            else:
                translation = (pred_sents, gold_sent, raw_src, query_id[b], pred_score[b])
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.pred_json_score_out_file = codecs.open(can_path+'.sample', 'w', 'utf-8')
        if self.args.dataset == 'paraphrase' and self.args.encoder == "roberta":
            self.pred_json_score_out_file.write("\t".join(["query_id", "source_query", "target_query", "predict_query"])+"\n")

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        cnt = 0
        pred_dict, ref_dict = {}, {}
        with torch.no_grad():
            for batch in data_iter:
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src, query_id, pred_score = trans
                    src = src.replace('<pad>', '').replace('##','').strip()
                    if self.args.dataset == "qg_ranking_test":
                        pred_str = "\t".join([each.replace('[unused0]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace('[SEP]', '').replace('[unused2]', '').replace(r' +', ' ').replace('<mask>', '<q>').replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('<unk>', ' ').strip() for each in pred])
                    else:
                        pred_str = pred.replace('[unused0]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace('[SEP]', '').replace('[unused2]', '').strip()
                        pred_str = pred_str.replace(r' +', ' ').replace('<mask>', '<q>').replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('<unk>', '').strip()
                    gold_str = gold.replace('<mask>', '<q>').strip().replace('UNK', '').replace('[unused1]', '').replace('[unused2]', '').replace('##', '').replace('<unk>', '')
                    if(self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str+ '<q>'+sent.strip()
                            can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                            # if(can_gap>=gap):
                            if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str


                    if self.args.dataset == "marco" or self.args.dataset == "squad" or self.args.dataset == "qg_ranking":
                        pred_str = pred_str.replace('<q>', ' ')
                        if query_id != None:
                            pred_json = {'query_id':query_id, 'answers':[pred_str]}
                            gold_json = {'query_id':query_id, 'answers':[gold_str]}
                            pred_json_score = {'query_id':query_id, 'answers':[pred_str], 'scores':pred_score[0].cpu().numpy().tolist()}
                        else:
                            pred_json = {'query_id':cnt, 'answers':[pred_str]}
                            gold_json = {'query_id':cnt, 'answers':[gold_str]}
                            pred_json_score = {'query_id':cnt, 'answers':[pred_str], 'scores':pred_score[0].cpu().numpy().tolist()}
                        json.dump(pred_json, self.can_out_file)
                        self.can_out_file.write('\n')
                        json.dump(gold_json, self.gold_out_file)
                        self.gold_out_file.write('\n')
                        json.dump(pred_json_score, self.pred_json_score_out_file)
                        self.pred_json_score_out_file.write('\n')
                        self.src_out_file.write(src.strip() + '\n')
                    elif self.args.dataset == "cnn":
                        self.can_out_file.write(pred_str + '\n')
                        self.gold_out_file.write(gold_str + '\n')
                        self.src_out_file.write(src.strip() + '\n')
                    elif self.args.dataset == "faq":
                        if pred_score[0].cpu().numpy().tolist() < -3.5:
                            continue
                        self.can_out_file.write("\t".join([str(query_id), src, pred_str])+"\n")
                        self.gold_out_file.write("\t".join([str(query_id), src, gold_str])+"\n")
                        # passage, answer, question, score
                        self.pred_json_score_out_file.write("\t".join([str(query_id), gold_str, src, pred_str, str(pred_score[0].cpu().numpy().tolist())])+"\n")
                    elif self.args.dataset == "qg_ranking_test":
                        self.can_out_file.write(str(query_id)+'\t'+pred_str+'\n') 
                     
                    cnt += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
        self.logger.info("cnt: %s" % cnt)
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            if self.args.dataset == "marco" or self.args.dataset == "squad" or self.args.dataset == "qg_ranking":
                cnn_results = subprocess.getoutput("./run.sh %s %s" % (gold_path, can_path))
                self.logger.info(cnn_results)
            elif self.args.dataset == "cnn":
                rouges = self._report_rouge(gold_path, can_path)
                self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                    self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                    self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(batch, self.max_length, min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        mask_src = batch.mask_src

        src_features = self.model.bert(src, None, mask_src)
        #src_features, _ = self.model.bert(src, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        if self.args.p_gen:
            src = tile(batch.src, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)

            dec_out, attns, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)

            # Generator forward.
            if self.args.p_gen:
                log_probs = self.generator.forward(src, dec_out, attns[-1], src_features)
            else:
                log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
               log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            #'''
            if self.args.sample_topk:
                temperature = self.args.temperature
                _scores = log_probs / temperature
                _scores = top_k_top_p_filtering(
                    _scores, top_k=self.args.top_k, top_p=self.args.top_p, min_tokens_to_keep=1
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                topk_ids = torch.multinomial(F.softmax(_scores, dim=-1), num_samples=1)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(_scores, dim=1)  # (batch_size * num_beams, vocab_size)
                
                _scores += topk_log_probs.view(-1).unsqueeze(1)
                _scores = _scores / length_penalty
                topk_scores = torch.gather(_scores, -1, topk_ids)  # (batch_size * num_beams, 2)
                #log_probs +=   # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                topk_ids = topk_ids.view(-1, beam_size)  # (batch_size, 2 * num_beams)
                topk_scores = topk_scores.view(-1, beam_size)  # (batch_size, 2 * num_beams)
            #'''
            else:
                curr_scores = log_probs / length_penalty

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if self.args.encoder == "roberta":
                            #words = [self.vocab.convert_ids_to_tokens[w] for w in words]
                            words = self.vocab.decode(words).strip().split()
                        else:
                            words = [self.vocab.ids_to_tokens[w] for w in words]
                            words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20
            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(self.end_token)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(self.end_token)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(self.end_token)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        if self.args.dataset == "qg_ranking_test" or (self.args.dataset == 'paraphrase' and (not self.args.sample_topk)): 
                            for each in best_hyp[:beam_size]:
                                score, pred = each
                                results["scores"][b].append(score)
                                results["predictions"][b].append(pred)
                        else:
                            score, pred = best_hyp[0]
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            if self.args.p_gen:
                src = src.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

def top_k_top_p_filtering(logits, top_k=10, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output

class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output

# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import types
from functools import reduce
import numpy as np

import torch
from torch import nn

logger = logging.getLogger(__name__)


class TextGenerator:
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
                 model,
                 vocab,
                 symbols,
                 beam_size=5,
                 min_length=0,
                 max_length=100,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.alpha = 0.6

        self.logger = logger
        # self.cuda = args.visible_gpus != '-1'
        self.cuda = (torch.cuda.device_count() > 0)

        self.model = model
        # TODO generator
        #self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['[CLS]'] #['[PAD]']
        self.end_token = symbols['[SEP]'] #'[PAD]']

        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.min_length = min_length
        self.max_length = max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

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

    def tile(self, x, count, dim=0):
        """
        Tiles x on dimension dim count times.
        """
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x
    
    def translate_batch(self, encoder_inputs, fast=False):
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
            return self._fast_translate_batch(encoder_inputs, self.max_length, min_length=self.min_length)

    def _fast_translate_batch(self,
                              encoder_inputs,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        tokens, types, padding_mask = encoder_inputs
        batch_size = tokens.size(0)
        device = tokens.device
        tmp_alive_seq = torch.full(
            [batch_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)
        prediction_scores, dec_feat_seq, sequence_output = self.model(tokens, types, padding_mask, tmp_alive_seq, None, None, checkpoint_activations=False, is_infer=True, sequence_output=None)
        src_features = sequence_output

        # Tile states and memory beam_size times.
        # dec_states.map_batch_fn(
        #     lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = self.tile(src_features, beam_size, dim=0)
        attention_mask = self.tile(padding_mask, beam_size, dim=0)
        #TODO support p_gen ...
        # if self.args.p_gen:
        #     src = tile(batch.src, beam_size, dim=0)
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
        results["batch"] = []
        dec_attn_mask = None
        dec_position_ids = None

        for step in range(max_length):
            tgt_len = alive_seq.size()[1]
            ## modified with tgt_len
            #repeat_attention_mask = attention_mask.repeat([1, 1, tgt_len, 1])
            #dec_attn_mask = _make_causal_mask(alive_seq.shape, attention_mask.dtype, device=alive_seq.device)
            #dec_feat_seq = self.model.decode(self.model.bert.embeddings, src_features, alive_seq,
            #                           enc_attn_mask=repeat_attention_mask, dec_attn_mask=dec_attn_mask)
            
            prediction_scores, dec_feat_seq, _ = self.model(tokens, types, attention_mask, alive_seq, dec_position_ids, dec_attn_mask, checkpoint_activations=False, is_infer=True, sequence_output=src_features)

            dec_feat_seq = dec_feat_seq[:, -1, :]
            vocab_size = dec_feat_seq.size(-1)
            log_probs = torch.log(torch.softmax(dec_feat_seq.view(-1, vocab_size), dim=-1))

            if step < min_length:
               log_probs[:, self.end_token] = -1e20
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.alpha #global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size, rounding_mode="trunc")
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
                is_finished.fill_(1) #self.end_token)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1) #self.end_token)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1) #self.end_token)
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
                        # if self.args.dataset == "qg_ranking_test" or (self.args.dataset == 'paraphrase' and (not self.args.sample_topk)):
                        #     for each in best_hyp[:beam_size]:
                        #         score, pred = each
                        #         results["scores"][b].append(score)
                        #         results["predictions"][b].append(pred)
                        # else:
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
            attention_mask = attention_mask.index_select(0, select_indices)

        return results


def run_generation_hf(pretrain_model_path,
                      output_dir,
                      task_type,
                      train_file_path=None,
                      dataset_name="text",
                      dev_file_path=None,
                      child_tuning_type=None,
                      reserve_p=0.2,
                      sequence_length=128,
                      target_length=100,
                      map_function=None,
                      filter_function=None,
                      save_strategy="steps",
                      save_total_limit=1,
                      seed=42,
                      config_args=None,
                      num_train_epochs=3,
                      **kwargs,
                      ):
    """ TODO
    Run a generation task with transformers code.
    :param pretrain_model_path: The local dir of pretrained model
    :param output_dir: The output directory where the model predictions and checkpoints will be written
    :param task_type: The task type, only support generation currently
    :param train_file_path: The local dir of train file
    :param dataset_name: The dataset name passed into datasets.load_dataset. Default will be "text"
    :param dev_file_path: The local dir of dev file, which is used in cross-validation in training.
    :param child_tuning_type: The child_tuning type. Can be "ChildTuning-F", "ChildTuning-D" or None
    :param reserve_p: the drop-out rate of child_tuning, default 0.2
    :param sequence_length: The max sequence length for padding
    :param target_length: The max target length for padding in generative task
    :param map_function: An optional map function, will be used with datasets.map()
    :param filter_function: An optional filter function, will be used with datasets.filter()
    :param save_strategy: TrainingArguments.
    :param save_total_limit: TrainingArguments.
    :param seed: Random seed, default 42.
    :param num_train_epochs: Total number of training epochs to perform, default 3.0.
    :param kwargs: Other optional hyper-parameters which is used in TrainingArguments.
    :return: None
    """

    # sofa custom code
    if config_args is None:
        config_args = {}
    from ... import environ
    environ("huggingface")
    # end
    from transformers import TrainingArguments
    from transformers import Trainer, set_seed
    import datasets
    from datasets import load_dataset, interleave_datasets
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import TrainerCallback,  TrainerState, TrainerControl
    from transformers.tokenization_utils import PreTrainedTokenizer
    # sofa dureader_eval
    from ...utils.dureader_eval import normalize, compute_bleu_rouge
    # sofa child_tuning
    from ...utils import apply_child_tuning_to_trainer

    cache_dir = ".cache"
    if "cache_dir" in kwargs:
        cache_dir = kwargs["cache_dir"]

    set_seed(seed)
    # assert type(label) in (str, types.FunctionType)
    logger.info(f"Cuda available:{torch.cuda.is_available()}")

    # prepare data
    train_datasets = None
    dev_datasets = None

    def get_files(files_or_path):
        if type(files_or_path) is str:
            if os.path.isfile(files_or_path):
                return files_or_path
            if os.path.isdir(files_or_path):
                return [os.path.join(files_or_path, f) for f in os.listdir(files_or_path)]
        if type(files_or_path) is list:
            return files_or_path

    if train_file_path is not None:
        train_files = get_files(train_file_path)
        data_files = {"train": train_files}
        train_datasets = load_dataset(dataset_name, split="train", data_files=data_files)
    if train_datasets is None:
        logger.error(f"dataset_name and train_file_path cannot both be None")

    if dev_file_path is not None:
        dev_files = get_files(dev_file_path)
        data_files = {"dev": dev_files}
        dev_datasets = load_dataset(dataset_name, split="dev", data_files=data_files)

    if filter_function is not None:
        train_datasets = train_datasets.filter(filter_function)
        if dev_datasets:
            dev_datasets = dev_datasets.filter(filter_function)
    if map_function is not None:
        train_datasets = train_datasets.map(map_function)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_function)

    if task_type != "generation":
        raise RuntimeError(f"Unsupported task type:{task_type}")

    def split_text(example):
        text_a, text_b = example[dataset_name].split('\t')
        return {"text_a": text_a, "text_b": text_b}
        
    train_datasets = train_datasets.map(split_text, remove_columns=["text"])
    if dev_datasets:
        dev_datasets = dev_datasets.map(split_text, remove_columns=["text"])
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path)

    kwargs_for_training = {}
    for p in TrainingArguments.__dataclass_fields__.keys():
        if p in kwargs:
            kwargs_for_training[p] = kwargs[p]
    kwargs_for_training["remove_unused_columns"] = False
    training_args = TrainingArguments(output_dir=output_dir,
                                      save_strategy=save_strategy,
                                      save_total_limit=save_total_limit,
                                      num_train_epochs=num_train_epochs,
                                      **kwargs_for_training)

    def tokenize_function(example):
        input_tokens = ["[CLS]"] + tokenizer.tokenize(example["text_a"]) + ["[SEP]"]
        if len(input_tokens) > sequence_length:
            input_tokens = input_tokens[:sequence_length-1] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        target_tokens = ["[CLS]"] + tokenizer.tokenize(example["text_b"]) + ["[SEP]"]
        if len(target_tokens) > target_length:
            target_tokens = target_tokens[:target_length-1] + ["[SEP]"]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        while len(target_ids) < target_length:
            target_ids.append(0)

        return {"input_ids": input_ids, "input_mask": input_mask,
                "segment_ids": segment_ids, "target_ids": target_ids}

    full_train_dataset = train_datasets.map(tokenize_function, remove_columns=["text_a", "text_b"])
    if dev_datasets:
        full_eval_dataset = dev_datasets.map(tokenize_function, remove_columns=["text_a", "text_b"])
    else:
        full_eval_dataset = None

    # Currently only supports palm model, better compatibility will be provided in the future
    class GeneratorCallback(TrainerCallback):
        def __init__(self, tokenizer: PreTrainedTokenizer, model: nn.Module, dataset: "datasets.Dataset", eval_batch_size: int = 16, num_workers: int = 1):
            self.tokenizer = tokenizer
            self.model = model
            self.eval_iters = len(dataset) // eval_batch_size
            sampler = torch.utils.data.SequentialSampler(dataset)
            self.data_loader = torch.utils.data.DataLoader(dataset,
                                                           batch_size=eval_batch_size,
                                                           sampler=sampler,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           collate_fn=model.palm_batchify)

        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            """Evaluation."""
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tokenizer, model, eval_iters, data_iterator = self.tokenizer, self.model, self.eval_iters, iter(self.data_loader)

            # Turn on evaluation mode which disables dropout.
            model.eval()

            pred_dict = {}
            ref_dict = {}
            vocab_size = 21127
            beam_generator = TextGenerator(model, tokenizer, tokenizer.vocab) 
            counter_all = 0

            with torch.no_grad():
                iteration = 0
                while iteration < eval_iters:
                    iteration += 1
                    # Forward evaluation.
                    # tokens, types, padding_mask, target_tokens, target_labels, dec_loss_mask, attention_mask, position_ids = self.get_batch(data_iterator, args, timers)
                    data = next(data_iterator)
                    tokens, types, padding_mask, target_labels = data["input_tokens"].to(device), data["token_type_ids"].to(device), data["attention_mask"].to(device), data["labels"]
                    batch_size = tokens.size(0)
                    # sequence_output = self.model.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False, checkpoint_activations=args.checkpoint_activations)
                    encoder_inputs = [tokens, types, padding_mask]

                    result_dict = beam_generator.translate_batch(encoder_inputs)
                    pred_list = result_dict["predictions"]
                    target_list = target_labels.cpu().numpy().tolist()
                    for i in range(batch_size):
                        pred_ids = pred_list[i][0].cpu().numpy().tolist()
                        for j in range(len(pred_ids)):
                            if pred_ids[j] > vocab_size-1:
                                pred_ids[j] = 100
                        gold = tokenizer.convert_ids_to_tokens(target_list[i])
                        pred = tokenizer.convert_ids_to_tokens(pred_ids)
                        gold_string = "".join(gold).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        pred_string = "".join(pred).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        if len(gold_string) < 3:
                            continue
                        pred_dict[str(counter_all)] = normalize([pred_string])
                        ref_dict[str(counter_all)] = normalize([gold_string])
                        counter_all += 1
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            print (ref_dict["0"])
            print (pred_dict["0"])
            print("test result %d: " % counter_all, bleu_rouge)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=model.palm_batchify,
                      train_dataset=full_train_dataset,
                      callbacks=[GeneratorCallback(tokenizer, model, full_eval_dataset)])

    # apply child_tuning or not.
    if child_tuning_type is not None:
        logger.info("Applying child-tuning.")
        apply_child_tuning_to_trainer(trainer, mode=child_tuning_type, reserve_p=reserve_p)
    trainer.train()

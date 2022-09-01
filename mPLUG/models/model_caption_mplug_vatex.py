from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class MPLUG(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.module_setting(config)
        self.visual_encoder, _ = initialize_clip(config)
        # self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        # self.fusion_encoder = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
        self.text_decoder = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)
        self.beam_generator = TextGenerator(config, self.text_decoder)

        self.prompt = config["prompt"]
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, quesiton, answer=None, train=True, out_size=5, scst=False, device=None):

        # shiyaya: for video
        B, N, C, W, H = image.size()
        image = image.view(-1, C, W, H)
        image = image.to(device, non_blocking=True)
        # end

        # if(scst):
        #     return self.beam_search(image, quesiton, answer, train=True,out_size=out_size)

        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))

        # # shiyaya: for video, temporal mean
        # C = image_embeds.size(-1)
        # image_embeds = image_embeds.view(B, N, -1, C).mean(dim=1)
        # # end

        # shiyaya: for video, concat
        C = image_embeds.size(-1)
        image_embeds = image_embeds.view(B, N, -1, C).view(B, -1, C)
        # end

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            if self.prompt != "":
                answer_targets[:, :self.prompt_length] = -100
            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=image_embeds,
                                              encoder_attention_mask=image_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )
            loss = answer_output.loss

            return loss


        else:
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            return topk_ids, topk_probs

    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True

    def beam_search(self, image, quesiton, answer=None, train=True, out_size=5):
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        if self.open_generation:
            topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size)
        else:
            topk_ids, topk_probs = self.rank_answer(question_output, quesiton.attention_mask if (
                        not self.merge_attention and not self.concat_last_layer) else merge_text_attention,
                                                    answer.input_ids, answer.attention_mask, k)
        return topk_ids, topk_probs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def generation(self, question_states, question_atts, answer_ids=None, answer_atts=None, k=None, out_size=1):
        input_ids = None
        if self.prompt != "":
            prompt = [self.prompt] * question_states.size(0)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(question_states.device)
            input_ids = input_ids[:, :-1]
        encoder_inputs = [question_states, question_atts, input_ids]
        topk_ids, topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs, out_size=out_size)
        return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

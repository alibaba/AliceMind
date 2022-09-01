# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for wrapping BertModel."""

import torch

from .modeling import BertConfig
from .modeling import PalmForPreTraining

class PalmModel(torch.nn.Module):

    def __init__(self, args):
        super(PalmModel, self).__init__()
        if args.pretrained_bert:
            self.model = BertForPreTraining.from_pretrained(
                args.tokenizer_model_type,
                cache_dir=args.cache_dir,
                fp32_layernorm=args.fp32_layernorm,
                fp32_embedding=args.fp32_embedding,
                layernorm_epsilon=args.layernorm_epsilon)
        else:
            if args.intermediate_size is None:
                intermediate_size = 4 * args.hidden_size
            else:
                intermediate_size = args.intermediate_size
            self.config = BertConfig(
                args.tokenizer_num_tokens,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=args.hidden_dropout,
                attention_probs_dropout_prob=args.attention_dropout,
                max_position_embeddings=args.max_position_embeddings,
                type_vocab_size=args.tokenizer_num_type_tokens,
                fp32_layernorm=args.fp32_layernorm,
                fp32_embedding=args.fp32_embedding,
                fp32_tokentypes=args.fp32_tokentypes,
                layernorm_epsilon=args.layernorm_epsilon,
                deep_init=args.deep_init,
                dec_hidden_layers=args.dec_layers)
            self.model = PalmForPreTraining(self.config)

    def forward(self, input_tokens, token_type_ids=None,
                attention_mask=None, target_tokens=None, position_ids=None, decode_attention_mask=None, checkpoint_activations=False, is_infer=False, sequence_output=None, parallel_output=True):
        return self.model(
            input_tokens, token_type_ids, attention_mask, target_tokens, position_ids, 
            decode_attention_mask, checkpoint_activations=checkpoint_activations, is_infer=is_infer, sequence_output=sequence_output, parallel_output=parallel_output)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)


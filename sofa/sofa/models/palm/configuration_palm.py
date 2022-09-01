# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import json
import copy

""" PALM model configuration """
from typing import Mapping
from collections import OrderedDict

from ...utils.configuration_utils import PretrainedConfig
from ...utils.modeling_utils import OnnxConfig
from ...utils.file_utils import logging


logger = logging.get_logger(__name__)


class PalmConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layernorm_epsilon (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        dec_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        attn_separate (:obj:`bool`, `optional`, defaults to false):
            Whether or not to separate the q, k, v of attention.

    Examples::

        >>> from sofa.models.palm import PalmModel, PalmConfig
        >>> configuration = PalmConfig()

        >>> # Initializing a model from the configuration
        >>> model = PalmModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type="palm"

    def __init__(
        self,
        vocab_size=21504,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=3,
        initializer_range=0.00707,
        lr_decay_style='linear',
        weight_decay=1e-2,
        clip_grad=1.0,
        warmup=0.01,
        pre_ln=True,
        layernorm_epsilon=1e-12,
        dec_hidden_layers=12,
        attn_separate=False,
        **kwargs
    ):
        super().__init__(layer_norm_eps=layernorm_epsilon, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.lr_decay_style = lr_decay_style
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.warmup = warmup
        self.pre_ln = pre_ln
        self.layernorm_epsilon = layernorm_epsilon
        self.dec_hidden_layers = dec_hidden_layers
        self.attn_separate = attn_separate
     
    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def merge_args(self, args):
        """merge values a `BertConfig` from a json file of parameters."""
        local_keys = self.__dict__.keys()
        for key, value in args.__dict__.items():
            if key in local_keys:
                continue
            self.__dict__[key] = value
        return self

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PalmOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

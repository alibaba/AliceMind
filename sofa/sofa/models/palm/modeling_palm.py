# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch PLAM model."""

import os
import math
import logging
import tarfile
import tempfile

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import checkpoint

from ...utils.modeling_utils import PreTrainedModel
from ...utils import ACT2FN

logger = logging.getLogger(__name__)

WEIGHTS_NAME = 'palm_model.pt'
# TF_WEIGHTS_NAME = 'palm_model.ckpt'


class PalmLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PalmEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings


class PalmSelfAttention(torch.nn.Module):
    """Self-attention layer for PALM.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size be divisible by n.
        dropout_prob: dropout probability for the attention scores.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 dropout_prob, separate = False):
        super().__init__()
        # Input configuration.
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.separate = separate
        # Per attention head
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        # Strided linear layer.
        if not separate:
            self.query_key_value = nn.Linear(hidden_size, 3*hidden_size)
        else:
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout = torch.nn.Dropout(dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, hidden_states, attention_mask):
        # Attention heads. [b, s, hp]
        if not self.separate:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = self._split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, np, s, s]
        norm_factor = math.sqrt(math.sqrt(self.hidden_size_per_attention_head))
        attention_scores = torch.matmul(query_layer/norm_factor,
                                        key_layer.transpose(-1, -2)/norm_factor)
        # Apply the attention mask.
        attention_scores += attention_mask

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Context layer.
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = context_layer

        return output


class PalmSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if not config.pre_ln:
            self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(ln_input)
        else:
            hidden_states = ln_input
        return hidden_states


class PalmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pre_ln:
            self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.self = PalmSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
            separate=config.attn_separate)
        self.output = PalmSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        if self.LayerNorm is not None:
            ln_input = input_tensor
            ln_output = self.LayerNorm(ln_input)
            self_output = self.self(ln_output, attention_mask)
        else:
            self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)

        return attention_output


class PalmIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class PalmOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if not config.pre_ln:
            self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        if self.LayerNorm is not None: 
            hidden_states = self.LayerNorm(ln_input)
        else:
            hidden_states = ln_input
        return hidden_states


class PalmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PalmAttention(config)
        self.intermediate = PalmIntermediate(config)
        self.output = PalmOutput(config)
        if config.pre_ln:
            self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        if self.LayerNorm is not None:
            ln_input = attention_output
            ln_output = self.LayerNorm(ln_input)
            intermediate_output = self.intermediate(ln_output)
        else:
            intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class PalmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([PalmLayer(config) for _ in range(config.num_hidden_layers)])
        if config.pre_ln:
            self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False, detach_index=-1):
        all_encoder_layers = []
        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = 1 #math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint(custom(l, l+chunk_length), hidden_states, attention_mask*1)
                if detach_index == l:
                    hidden_states.detach_()
                l += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
                if detach_index == i:
                    hidden_states.detach_()
                if i == len(self.layer) - 1 and self.LayerNorm is not None:
                    hidden_states = self.LayerNorm(hidden_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if self.LayerNorm is not None:
                hidden_states = self.LayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class PalmPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PalmPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class PalmLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = PalmPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = F.linear(hidden_states, self.decoder_weight, self.bias)

        return hidden_states


class PalmPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = PalmLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 3)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        for p in self.seq_relationship.parameters():
            if p is None:
                continue
            pooled_output = pooled_output.type_as(p)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PalmPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, PalmLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, strict=True, *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a path or url to a pretrained model archive containing:
                    . `config.json` a configuration file for the model
                    . `palm_model.pt` a palm model checkpoint
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        logger.info("loading archive file {}".format(pretrained_model_name))
        resolved_archive_file = pretrained_model_name
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, tempdir)
            serialization_dir = tempdir
        # Load config
        # config_file = os.path.join(serialization_dir, CONFIG_NAME)
        # config = PalmConfig.from_json_file(config_file) if os.path.isfile(config_file) else PalmConfig()
        # config.layernorm_epsilon = layernorm_epsilon
        # logger.info("Model config {}".format(config))
        # Instantiate model.
        # model = cls(config, *inputs, **kwargs)
        model = cls(*inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict, strict=strict)

        return model


class PalmEncoderModel(PalmPreTrainedModel):
    """PALM encoder model
    Params:
        config: a PalmConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = PalmEmbeddings(config)
        self.encoder = PalmEncoder(config)
        self.pooler = PalmPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, checkpoint_activations=False, detach_index=-1):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.encoder.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      checkpoint_activations=checkpoint_activations,
                                      detach_index=detach_index)
        sequence_output = encoded_layers[-1]
        for p in self.pooler.parameters():
            if p is None:
                continue
            sequence_output = sequence_output.type_as(p)
            break
        #pooled_output = self.pooler(sequence_output)
        pooled_output = sequence_output[:, 0]
        if not output_all_encoded_layers or checkpoint_activations:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class PalmDecoderSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # Per attention head
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        # Strided linear layer.
        self.query_key_value = nn.Linear(hidden_size, 3*hidden_size)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, hidden_states, ltor_mask, is_infer=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        tgt_len = hidden_states.size(1)
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = self._split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        
        previous_type = value_layer.type()

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        # Apply the left to right attention mask.
        if is_infer:
            src_len = key_layer.size(2)
            ltor_mask = torch.tril(torch.ones(
                        (1, tgt_len, src_len), device=hidden_states.device)).view(1, 1, tgt_len, src_len).type(previous_type)
        attention_scores = torch.mul(attention_scores, ltor_mask) - \
                           10000.0 * (1.0 - ltor_mask)

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class PalmDecoderCrossAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 attn_separate=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # Per attention head
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        # Strided linear layer.
        self.query = nn.Linear(hidden_size, hidden_size)
        
        if not attn_separate:
            self.key_value = nn.Linear(hidden_size, 2*hidden_size)
        else:
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        self.attn_separate = attn_separate

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, query, enc_hidden_states, enc_attn_mask):
        # Attention heads. [b, s, hp]
        mixed_query_layer = self.query(query)
        #print_rank_0(enc_hidden_states.size())
        if not self.attn_separate:
            mixed_x_layer = self.key_value(enc_hidden_states)
            (mixed_key_layer, mixed_value_layer) = self._split_tensor_along_last_dim(mixed_x_layer, 2)
        else:
            mixed_key_layer = self.key(enc_hidden_states)
            mixed_value_layer = self.value(enc_hidden_states)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        attention_scores += enc_attn_mask

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class PalmDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PalmDecoderSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout_prob=config.attention_probs_dropout_prob,
            output_dropout_prob=config.hidden_dropout_prob)

        self.cross_attention = PalmDecoderCrossAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout_prob=config.attention_probs_dropout_prob,
            output_dropout_prob=config.hidden_dropout_prob,
            attn_separate=False)
        
        self.input_layernorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_cross_attention_layernorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, enc_hidden_states, enc_attn_mask, dec_attn_mask, is_infer=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, dec_attn_mask, is_infer=is_infer)
        # add dropout?
        hidden_states = residual + hidden_states

        residual = hidden_states     
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.cross_attention(hidden_states, enc_hidden_states, enc_attn_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class PalmDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([PalmDecoderLayer(config) for _ in range(config.dec_hidden_layers)])
        
        self.final_layernorm = PalmLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states, enc_hidden_states, enc_attn_mask, dec_attn_mask, checkpoint_activations=False, is_infer=False):
        pre_enc_hidden= enc_hidden_states.data
        #pre_enc_hidden= enc_hidden_states.clone()
        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = 1 #math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                #enc_hidden_states = pre_enc_hidden.clone()
                enc_hidden_states.data = pre_enc_hidden
                l += chunk_length
            # decoder layers
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, enc_hidden_states, enc_attn_mask, dec_attn_mask, is_infer=is_infer)
        
        hidden_states = self.final_layernorm(hidden_states)
        
        return [hidden_states]


class PalmDecoderModel(PalmPreTrainedModel):
    """PALM decoder model

    Params:
        config: a PalmConfig class instance with the configuration to build a new model

    Inputs:
        `embeddings`: same embeddings as encoder.
        `sequence_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
        `decode_input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `enc_attn_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `dec_attn_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. Similar to enc_attn_mask.

    Outputs: pooled_output(at the last position of sequence_output)
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    """
    def __init__(self, config):
        super().__init__(config)
        self.decoder = PalmDecoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, embeddings, sequence_output, decode_input_ids, position_ids=None, enc_attn_mask=None, dec_attn_mask=None, checkpoint_activations=False, is_infer=False):

        extended_attention_mask = enc_attn_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.decoder.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = embeddings(decode_input_ids)
        sequence_output = self.decoder(embedding_output,
                                       sequence_output,
                                       extended_attention_mask,
                                       dec_attn_mask,
                                       checkpoint_activations=checkpoint_activations,
                                       is_infer=is_infer)

        return sequence_output[-1]


class PalmModel(PalmPreTrainedModel):
    """Palm model.
    The palm model consists of an encoder model, a pretrainedhead and a decoder model.

    Params:
        config: a PalmConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `decode_input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `decode_attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `is_infer`: a boolean indicating whether the model is used for inference.

    Outputs:
        if `is_infer` is `True`:
            Tuple of (preduction_scores, seq_relationship_logits, encoder_sequence_output).
        if `is_infer` is `False`:
            Tuple if (preduction_scores, seq_relationship_logits).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = PalmConfig(vocab_size_or_config_json_file=21504, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = PalmModel(config)
    preduction_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.bert = PalmEncoderModel(config)
        self.cls = PalmPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.decoder = PalmDecoderModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                decode_input_ids=None, position_ids=None, decode_attention_mask=None,
                lm_labels=None, checkpoint_activations=False, is_infer=False, sequence_output=None):
        
        if sequence_output is None:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                       output_all_encoded_layers=False, checkpoint_activations=checkpoint_activations)
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        else:
            prediction_scores = None
            seq_relationship_score = None
            sequence_output = sequence_output.to(dtype=next(self.decoder.parameters()).dtype)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        decode_output = self.decoder(self.bert.embeddings, sequence_output, decode_input_ids, position_ids, attention_mask,
                                     decode_attention_mask, checkpoint_activations=checkpoint_activations, is_infer=is_infer)

        #prediction_scores = self.cls(decode_output)
        
        logits = F.linear(decode_output, self.bert.embeddings.word_embeddings.weight)
        
        if is_infer:
            return prediction_scores, logits, sequence_output
        return prediction_scores, logits


class PalmForConditionalGeneration(PalmPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = PalmModel(config)

    def forward(self, input_tokens, token_type_ids=None, attention_mask=None,
                target_tokens=None, position_ids=None, decode_attention_mask=None,
                checkpoint_activations=False, is_infer=False, sequence_output=None,
                labels=None, dec_loss_mask=None):
        output = self.model(input_tokens, token_type_ids, attention_mask, target_tokens,
                            position_ids, decode_attention_mask, checkpoint_activations=checkpoint_activations,
                            is_infer=is_infer, sequence_output=sequence_output)

        if is_infer:
            return output
        _, logits = output
        _logits = logits.view(-1, logits.size()[-1]).contiguous().float()
        losses = F.cross_entropy(_logits, labels.view(-1).contiguous())
        dec_loss_mask = dec_loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * dec_loss_mask) / dec_loss_mask.sum()

        return loss, logits

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def palm_batchify(data):
        tmp = {k: list() for k in data[0]}
        for dataset in data:
            for k, v in dataset.items():
                tmp[k].append(torch.tensor(v))
        data = {k: torch.stack(v, 0) for k, v in tmp.items()}

        tokens = data['input_ids'].long()
        types = data['segment_ids'].long()
        padding_mask = data['input_mask'].byte()
        target_ids = data['target_ids'].long()

        target_tokens = target_ids[:, :-1].contiguous()
        target_labels = target_ids[:, 1:].contiguous()

        def get_masks_and_position_ids(data, eod_token):
            _, seq_length = data.size()
            # Attention mask (lower triangular).
            att_mask_batch = 1
            attention_mask = torch.tril(torch.ones(
                (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                    att_mask_batch, 1, seq_length, seq_length)
            # Loss mask.
            loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
            loss_mask[data == eod_token] = 0.0
            # Position ids.
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=data.device)
            position_ids = position_ids.unsqueeze(0).expand_as(data).clone()

            return attention_mask, loss_mask, position_ids       

        # Get the masks and postition ids.
        attention_mask, dec_loss_mask, position_ids = get_masks_and_position_ids(target_tokens, 0)

        return {"input_tokens": tokens, "token_type_ids": types, "attention_mask": padding_mask,
                "target_tokens": target_tokens, "position_ids": position_ids, "decode_attention_mask": attention_mask,
                "labels": target_labels, "dec_loss_mask": dec_loss_mask}

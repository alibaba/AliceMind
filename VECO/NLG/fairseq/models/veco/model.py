# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation
"""

import logging
import re

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import VECOHubInterface


logger = logging.getLogger(__name__)


@register_model('veco')
class VECOModel(TransformerModel):

    @classmethod
    def hub_models(cls):
        return {  # todo: update link
            'veco.large': 'http://dl.fbaipublicfiles.com/fairseq/models/veco.large.tar.gz',
            'veco.large.wmt14': 'http://dl.fbaipublicfiles.com/fairseq/models/veco.large.mnli.tar.gz',
            'veco.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/veco.large.cnn.tar.gz',
            'veco.large.xsum': 'http://dl.fbaipublicfiles.com/fairseq/models/veco.large.xsum.tar.gz',
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        super(VECOModel, VECOModel).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )
        parser.add_argument(
            '--keep-encoder-layers', type=str, default=None,
            help='If set, only keep part of encoder layers'
        )
        parser.add_argument(
            '--keep-decoder-layers', type=str, default=None,
            help='If set, only keep part of decoder layers'
        )
    @property
    def supported_targets(self):
        return {'self'}

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return VECOHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = VECOClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        keys_to_delete = []
        keys_to_reset = {}
        # Delete layers which are not in current state_dict
        if self.args.keep_encoder_layers is not None:
            loaded_encoder_keys = []
            loaded_encoder_layers = []
            for key in state_dict.keys():
                if "encoder.layers" in key:
                    layer_i = int(re.findall('.*?layers\.(.*?)\..*?', key)[0])
                    loaded_encoder_layers.append(layer_i)
                    loaded_encoder_keys.append(key)
            keep_encoder_layers = [int(i) for i in self.args.keep_encoder_layers.split(',')]
            if len(set(loaded_encoder_layers)) > len(keep_encoder_layers):
                for layer_i, key in zip(loaded_encoder_layers, loaded_encoder_keys):
                    if layer_i not in keep_encoder_layers:
                        logger.info('Deleting encoder: {}'.format(key))
                        keys_to_delete.append(key)
                    else:
                        new_key = key.replace(str(layer_i), str(keep_encoder_layers.index(layer_i)))
                        if new_key != key:
                            logger.info('Change layer from {} to {}'.format(key, new_key))
                            keys_to_reset[key] = new_key
        if self.args.keep_decoder_layers is not None:
            loaded_decoder_keys = []
            loaded_decoder_layers = []
            for key in state_dict.keys():
                if "decoder.layers" in key:
                    layer_i = int(re.findall('.*?layers\.(.*?)\..*?', key)[0])
                    loaded_decoder_layers.append(layer_i)
                    loaded_decoder_keys.append(key)
            logger.info('Loaded layers: {}'.format(','.join([str(i) for i in set(loaded_decoder_layers)])))
            logger.info('Kept   layers: {}'.format(self.args.keep_decoder_layers))
            keep_decoder_layers = [int(i) for i in self.args.keep_decoder_layers.split(',')]
            if len(set(loaded_decoder_layers)) > len(keep_decoder_layers):
                for layer_i, key in zip(loaded_decoder_layers, loaded_decoder_keys):
                    if layer_i not in keep_decoder_layers:
                        logger.info('Deleting decoder: {}'.format(key))
                        keys_to_delete.append(key)
                    else:
                        new_key = key.replace(str(layer_i), str(keep_decoder_layers.index(layer_i)))
                        if new_key != key:
                            logger.info('Change layer from {} to {}'.format(key, new_key))
                            keys_to_reset[key] = new_key

        for k in keys_to_delete:
            del state_dict[k]
        for k1, k2 in keys_to_reset.items():
            state_dict[k2] = state_dict[k1]
            del state_dict[k1]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        logger.info('!!!!!! loaded_dict_size: {}'.format(loaded_dict_size))
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            logger.info('Remove last row of embedding matrix that corresponds to mask_idx token.')
            truncate_emb('encoder.embed_tokens.weight')
            truncate_emb('decoder.embed_tokens.weight')
            truncate_emb('encoder.output_projection.weight')
            truncate_emb('decoder.output_projection.weight')

        # We resized the dictonary when fine-tuning on a specified pair of language.
        # Thus we remove the lang embeddings which are not in the resized dictionary.
        if self.args.task == 'translation_from_pretrained_veco' and loaded_dict_size > len(self.encoder.dictionary) + 1:
            def resized_all_emb(bool_keep):
                for key in ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight',
                            'encoder.output_projection.weight', 'decoder.output_projection.weight']:
                    if key in state_dict:
                        logger.info("Resize `{}`".format(key))
                        state_dict[key] = state_dict[key][bool_keep, :]

            logger.info("*** Resized the dictonary when fine-tuning on a specified pair of language. ***")
            assert self.args.resized_dict_map, \
                "--resized-dict-map must be set if the size of dictionary: {}+1 " \
                "is the not the same as state_dict['encoder.embed_tokens.weight'].size(0):{}".format(
                    len(self.encoder.dictionary), state_dict['encoder.embed_tokens.weight'].size(0))

            # The 0-th word in dict.txt start at index 4 in fairseq vocab
            # Vocab    |    0    |    1    |   2    |    3    |  4    |  5  |  6  |   7   |   8   |  9
            # -------- | ------- | ------- | ------ | ------- | ----- | --- | --- | ----- | ----- | ----
            # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ','   | '.' | '▁' | 's'   | '▁de' | '-'
            # dict.txt |  ','    |   '.'   |   '▁'  |   's'   | '▁de' | '-' | '▁a'
            try:
                bool_keep = [True, True, True, True]
                dict_map = open(self.args.resized_dict_map, "r", encoding="utf-8").read().split('\n')
                if dict_map[-1] == "":  # last line is blank
                    dict_map = dict_map[:-1]
                logger.info("The size of resized_dict_map is: {}".format(len(dict_map)))
                bool_keep.extend([bool(int(i)) for i in dict_map])
                if '<mask>' not in self.encoder.dictionary:
                    bool_keep.append(False)  # remove the embedding of <mask>
                else:
                    bool_keep.append(True)  # keep the embedding of <mask>
                assert len(bool_keep) == loaded_dict_size, \
                    "The size of resized_dict_map: {}+5 != \n " \
                    "The size of state_dict['encoder.embed_tokens.weight']: {}".format(len(dict_map), loaded_dict_size)
                resized_all_emb(bool_keep)
                new_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
                logger.info("*** Successfully resize or truncate the embeeding matrix, now dict_size is:{}, "
                            "removed rows: {} ***".format(new_dict_size, len(bool_keep) - new_dict_size))
            except Exception as e:
                raise e

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v
            return state_dict


class VECOClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture('veco', 'veco_large')
def veco_large_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.keep_decoder_layers = getattr(args, 'keep_decoder_layers', None)
    if args.keep_decoder_layers is not None:
        args.decoder_layers = len(args.keep_decoder_layers.split(','))
    else:
        args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 512)
    args.max_source_positions = getattr(args, 'max_source_positions', 512)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('veco', 'veco_xlarge')
def veco_xlarge_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4 * 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.max_target_positions = getattr(args, 'max_target_positions', 512)
    args.max_source_positions = getattr(args, 'max_source_positions', 512)
    veco_large_architecture(args)


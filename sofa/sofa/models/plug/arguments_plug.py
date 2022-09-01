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

"""argparser configuration"""

import argparse
import os
import json
import torch
import deepspeed
from sofa.utils import ArgsBase, print_rank_0

class PlugArgs(ArgsBase):

    def __init__(self, parser=None):
        super().__init__(parser)
        if parser == None:
            self.parser = ArgsBase._get_default_parser()  

    def _add_model_config_args(self):
        """Model arguments"""

        group = self.parser.add_argument_group('model', 'model configuration')

        group.add_argument('--pretrained-bert', action='store_true',
                        help='use a pretrained bert-large-uncased model instead'
                        'of initializing from scratch. See '
                        '--tokenizer-model-type to specify which pretrained '
                        'BERT model to use')
        group.add_argument('--attention-dropout', type=float, default=0.1,
                        help='dropout probability for attention weights')
        group.add_argument('--num-attention-heads', type=int, default=16,
                        help='num of transformer attention heads')
        group.add_argument('--hidden-size', type=int, default=1024,
                        help='tansformer hidden size')
        group.add_argument('--intermediate-size', type=int, default=None,
                        help='transformer embedding dimension for FFN'
                        'set to 4*`--hidden-size` if it is None')
        group.add_argument('--num-layers', type=int, default=24,
                        help='num decoder layers')
        group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                        help='layer norm epsilon')
        group.add_argument('--hidden-dropout', type=float, default=0.1,
                        help='dropout probability for hidden state transformer')
        group.add_argument('--max-position-embeddings', type=int, default=512,
                        help='maximum number of position embeddings to use')
        group.add_argument('--vocab-size', type=int, default=30522,
                        help='vocab size to use for non-character-level '
                        'tokenization. This value will only be used when '
                        'creating a tokenizer')
        group.add_argument('--deep-init', action='store_true',
                        help='initialize bert model similar to gpt2 model.'
                        'scales initialization of projection layers by a '
                        'factor of 1/sqrt(2N). Necessary to train bert '
                        'models larger than BERT-Large.')
        group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                        help='Pad the vocab size to be divisible by this value.'
                        'This is added for computational efficieny reasons.')
        group.add_argument('--cpu-optimizer', action='store_true',
                                    help='Run optimizer on CPU')
        group.add_argument('--cpu_torch_adam', action='store_true',
                                    help='Use Torch Adam as optimizer on CPU.')

        return self.parser

    def _add_fp16_config_args(self):
        """Mixed precision arguments."""

        group = self.parser.add_argument_group('fp16', 'fp16 configurations')

        group.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode')
        group.add_argument('--fp32-embedding', action='store_true',
                        help='embedding in fp32')
        group.add_argument('--fp32-layernorm', action='store_true',
                        help='layer norm in fp32')
        group.add_argument('--fp32-tokentypes', action='store_true',
                        help='embedding token types in fp32')
        group.add_argument('--fp32-allreduce', action='store_true',
                        help='all-reduce in fp32')
        group.add_argument('--hysteresis', type=int, default=2,
                        help='hysteresis for dynamic loss scaling')
        group.add_argument('--loss-scale', type=float, default=None,
                        help='Static loss scaling, positive power of 2 '
                        'values can improve fp16 convergence. If None, dynamic'
                        'loss scaling is used.')
        group.add_argument('--loss-scale-window', type=float, default=1000,
                        help='Window over which to raise/lower dynamic scale')
        group.add_argument('--min-scale', type=float, default=1,
                        help='Minimum loss scale for dynamic loss scale')

        return self.parser

    def _add_training_args(self):
        """Training arguments."""

        group = self.parser.add_argument_group('train', 'training configurations')

        group.add_argument('--batch-size', type=int, default=32,
                        help='Data Loader batch size')
        group.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay coefficient for L2 regularization')
        group.add_argument('--checkpoint-activations', action='store_true',
                        help='checkpoint activation to allow for training '
                        'with larger models and sequences')
        group.add_argument('--checkpoint-num-layers', type=int, default=1,
                        help='chunk size (number of layers) for checkpointing')
        group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                        help='uses activation checkpointing from deepspeed')
        group.add_argument('--clip-grad', type=float, default=1.0,
                        help='gradient clipping')
        group.add_argument('--train-iters', type=int, default=1000000,
                        help='total number of iterations to train over all training runs')
        group.add_argument('--log-interval', type=int, default=5,
                        help='report interval')
        group.add_argument('--exit-interval', type=int, default=None,
                        help='Exit the program after this many new iterations.')

        group.add_argument('--seed', type=int, default=1234,
                        help='random seed')
        # Batch prodecuer arguments
        group.add_argument('--reset-position-ids', action='store_true',
                        help='Reset posistion ids after end-of-document token.')
        group.add_argument('--reset-attention-mask', action='store_true',
                        help='Reset self attention maske after '
                        'end-of-document token.')

        # Learning rate.
        group.add_argument('--lr-decay-iters', type=int, default=None,
                        help='number of iterations to decay LR over,'
                        ' If None defaults to `--train-iters`*`--epochs`')
        group.add_argument('--lr-decay-style', type=str, default='linear',
                        choices=['constant', 'linear', 'cosine', 'exponential'],
                        help='learning rate decay function')
        group.add_argument('--lr', type=float, default=3e-5,
                        help='initial learning rate')
        group.add_argument('--warmup', type=float, default=0.01,
                        help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')
        group.add_argument('--batch-warmup', type=float, default=0.01,
                        help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')                       
        group.add_argument('--length-warmup', type=float, default=0.01,
                        help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')
        # model checkpointing
        group.add_argument('--save', type=str, default=None,
                        help='Output directory to save checkpoints to.')
        group.add_argument('--save-interval', type=int, default=2000,
                        help='number of iterations between saves')
        group.add_argument('--no-save-optim', action='store_true',
                        help='Do not save current optimizer.')
        group.add_argument('--no-save-rng', action='store_true',
                        help='Do not save current rng state.')
        group.add_argument('--load', type=str, default=None,
                        help='Path to a directory containing a model checkpoint.')
        group.add_argument('--load-iteration', type=str, default=0,
                        help='Load iteration of a model checkpoint.')
        group.add_argument('--pre-load', action='store_true',
                        help='Use pre-load instead of deepspeed load.')
        group.add_argument('--no-load-optim', action='store_true',
                        help='Do not load optimizer when loading checkpoint.')
        group.add_argument('--no-load-rng', action='store_true',
                        help='Do not load rng state when loading checkpoint.')
        group.add_argument('--no-load-lr', action='store_true',
                        help='Do not load lr schedule when loading checkpoint.')    
        group.add_argument('--finetune', action='store_true',
                        help='Load model for finetuning. Do not load optimizer '
                        'or rng state from checkpoint and set iteration to 0. '
                        'Assumed when loading a release checkpoint.')
        group.add_argument('--resume-dataloader', action='store_true',
                        help='Resume the dataloader when resuming training. '
                        'Does not apply to tfrecords dataloader, try resuming'
                        'with a different seed in this case.')
        # distributed training args
        group.add_argument('--distributed-backend', default='nccl',
                        help='which backend to use for distributed '
                        'training. One of [gloo, nccl]')

        group.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher')

        return self.parser

    def _add_evaluation_args(self):
        """Evaluation arguments."""

        group = self.parser.add_argument_group('validation', 'validation configurations')

        group.add_argument('--eval-batch-size', type=int, default=None,
                        help='Data Loader batch size for evaluation datasets.'
                        'Defaults to `--batch-size`')
        group.add_argument('--eval-iters', type=int, default=100,
                        help='number of iterations to run for evaluation'
                        'validation/test for')
        group.add_argument('--eval-interval', type=int, default=1000,
                        help='interval between running evaluation on validation set')
        group.add_argument('--eval-seq-length', type=int, default=None,
                        help='Maximum sequence length to process for '
                        'evaluation. Defaults to `--seq-length`')
        group.add_argument('--eval-max-preds-per-seq', type=int, default=None,
                        help='Maximum number of predictions to use for '
                        'evaluation. Defaults to '
                        'math.ceil(`--eval-seq-length`*.15/10)*10')
        group.add_argument('--overlapping-eval', type=int, default=32,
                        help='sliding window for overlapping eval ')
        group.add_argument('--cloze-eval', action='store_true',
                        help='Evaluation dataset from `--valid-data` is a cloze task')
        group.add_argument('--eval-hf', action='store_true',
                        help='perform evaluation with huggingface openai model.'
                        'use `--load` to specify weights path to be loaded')
        group.add_argument('--load-openai', action='store_true',
                        help='load openai weights into our model. Use `--load` '
                        'to specify weights path to be loaded')

        return self.parser

    def _add_text_generate_args(self):
        """Text generate arguments."""

        group = self.parser.add_argument_group('Text generation', 'configurations')
        group.add_argument("--temperature", type=float, default=1.0)
        group.add_argument("--top_p", type=float, default=0.0)
        group.add_argument("--top_k", type=int, default=0)
        group.add_argument("--out-seq-length", type=int, default=256)
        return self.parser

    def _add_struct_args(self):
        group = self.parser.add_argument_group('struct', 'struct configurations')
        group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help='Not Imp yet.')
        group.add_argument("--num-epochs", type=int, default=30)
        group.add_argument("--struct-bert-dataset", action='store_true', default=False,
                        help='Use struct bert dataset or not.')
        return self.parser

    def _add_distil_args(self):
        group = self.parser.add_argument_group('distil', 'distil configurations')
        group.add_argument("--bert-config-file", type=str, default=None,
                        help="teacher configs that define teacher model size.")
        group.add_argument("--distil-temperature", type=float, default=1.0,
                        help="temperature of softmax")
        group.add_argument("--environ", type=str, default='local', help='jiuding for oss local')
        group.add_argument('--teacher-load', type=str, default=None,
                        help='Path to a directory containing a model checkpoint.')
        group.add_argument('--teacher-load-iteration', type=str, default=0,
                        help='Load iteration of a model checkpoint.')
        return self.parser

    def _add_palm_args(self):
        group = self.parser.add_argument_group('palm', 'struct configurations')
        group.add_argument('--dec-layers', type=int, default=6,
                        help='num decoder layers')
        group.add_argument('--tgt-length', type=int, default=100,
                        help='num decoder layers')
        group.add_argument('--min-length', type=int, default=1,
                        help='num decoder layers')
        group.add_argument('--beam-size', type=int, default=5,
                        help='beam')
        group.add_argument("--sample-topk", action='store_true', default=False,
                        help='Use struct bert dataset or not.')
        group.add_argument('--vae-size', type=int, default=8192,
                        help='vae code vocab size')
        group.add_argument('--max-image-position', type=int, default=1025,
                        help='max image decode position')
        group.add_argument('--palm-dataset', action='store_true', default=False,
                        help='Use struct bert dataset or not.')
        group.add_argument('--image-dataset', action='store_true', default=False,
                        help='Use struct bert dataset or not.')
        group.add_argument('--do-mask-lm', action='store_true', default=False,
                        help='Do mask lm task or not.')
        group.add_argument('--vae-enc-model', type=str, default=None,
                        help='Path to a directory containing a model checkpoint.')
        group.add_argument('--vae-dec-model', type=str, default=None,
                        help='Path to a directory containing a model checkpoint.')
        return self.parser

    def _add_downstream_args(self):
        group = self.parser.add_argument_group('downstream', 'struct configurations')
        group.add_argument('--downstream-dataset', action='store_true', default=False,
                        help='Use struct bert dataset or not.')
        group.add_argument('--task-name', default='ocnli', type=str)
        group.add_argument('--detach-index', default='-1', type=int)
        return self.parser

    def _add_data_args(self):
        """Train/valid/test data arguments."""

        group = self.parser.add_argument_group('data', 'data configurations')

        group.add_argument('--model-parallel-size', type=int, default=1,
                        help='size of the model parallel.')
        group.add_argument('--shuffle', action='store_true',
                        help='Shuffle data. Shuffling is deterministic '
                        'based on seed and current epoch.')
        group.add_argument('--train-data', default=None,
                        help='Filename or corpora name for training.')
        group.add_argument('--dev-data', default=None)

        group.add_argument('--use-npy-data-loader', action='store_true',
                        help='Use the numpy data loader. If set, then'
                        'train-data-path, val-data-path, and test-data-path'
                        'should also be provided.')
        group.add_argument('--train-data-path', type=str, default='',
                        help='path to the training data')
        group.add_argument('--val-data-path', type=str, default='',
                        help='path to the validation data')
        group.add_argument('--test-data-path', type=str, default='',
                        help='path to the test data')
        group.add_argument('--input-data-sizes-file', type=str, default='sizes.txt',
                        help='the filename containing all the shards sizes')

        group.add_argument('--delim', default=',',
                        help='delimiter used to parse csv data files')
        group.add_argument('--text-key', default='sentence',
                        help='key to use to extract text from json/csv')
        group.add_argument('--eval-text-key', default=None,
                        help='key to use to extract text from '
                        'json/csv evaluation datasets')
        group.add_argument('--valid-data', nargs='*', default=None,
                        help="""Filename for validation data.""")
        group.add_argument('--split', default='1000,1,1',
                        help='comma-separated list of proportions for training,'
                        ' validation, and test split')
        group.add_argument('--test-data', nargs='*', default=None,
                        help="""Filename for testing""")

        group.add_argument('--lazy-loader', action='store_true',
                        help='whether to lazy read the data set')
        group.add_argument('--loose-json', action='store_true',
                        help='Use loose json (one json-formatted string per '
                        'newline), instead of tight json (data file is one '
                        'json string)')
        group.add_argument('--presplit-sentences', action='store_true',
                        help='Dataset content consists of documents where '
                        'each document consists of newline separated sentences')
        group.add_argument('--num-workers', type=int, default=2,
                        help="""Number of workers to use for dataloading""")
        group.add_argument('--tokenizer-model-type', type=str,
                        default='bert-large-uncased',
                        help="Model type to use for sentencepiece tokenization \
                        (one of ['bpe', 'char', 'unigram', 'word']) or \
                        bert vocab to use for BertWordPieceTokenizer (one of \
                        ['bert-large-uncased', 'bert-large-cased', etc.])")
        group.add_argument('--tokenizer-path', type=str, default='tokenizer.model',
                        help='path used to save/load sentencepiece tokenization '
                        'models')
        group.add_argument('--tokenizer-type', type=str,
                        default='BertWordPieceTokenizer',
                        choices=['CharacterLevelTokenizer',
                                    'SentencePieceTokenizer',
                                    'BertWordPieceTokenizer',
                                    'GPT2BPETokenizer'],
                        help='what type of tokenizer to use')
        group.add_argument('--cache-dir', default=None, type=str,
                        help='Where to store pre-trained BERT downloads')
        group.add_argument('--use-tfrecords', action='store_true',
                        help='load `--train-data`, `--valid-data`, '
                        '`--test-data` from BERT tf records instead of '
                        'normal data pipeline')
        group.add_argument('--seq-length', type=int, default=128,
                        help='Maximum sequence length to process')
        group.add_argument('--max-preds-per-seq', type=int, default=None,
                        help='Maximum number of predictions to use per sequence.'
                        'Defaults to math.ceil(`--seq-length`*.15/10)*10.'
                        'MUST BE SPECIFIED IF `--use-tfrecords` is True.')

        return self.parser

    def _add_pruning_args(self):
        group = self.parser.add_argument_group('pruning', 'pruning configurations')

        # group.add_argument('--pruning', action='store_true', help='pruning')  # TODO(note): Already delete this augment
        group.add_argument('--pruning-method', default=None,
                        #    choices=['topK', 'threshold', 'sigmoid_threshold', 'magnitude', 'l0', 'finetune', 'taylor', 'taylor_complete'],
                        help='what type of pruning method to use')
        group.add_argument('--pruning-mask-init', default='constant', type=str,
                        help='Initialization method for the mask scores. Choices: constant, uniform, kaiming.')
        group.add_argument('--pruning-mask-scale', default=0.0, type=float,
                        help='Initialization parameter for the chosen initialization method.')
        group.add_argument('--pruning-mask-scores-learning-rate', default=0.1, type=float,
                        help='The Adam initial learning rate of the mask scores.')
        group.add_argument('--pruning-initial-threshold', default=1.0, type=float,
                        help='Initial value of the threshold (for scheduling).')
        group.add_argument('--pruning-final-threshold', default=0.7, type=float,
                        help='Final value of the threshold (for scheduling).')
        group.add_argument('--pruning-initial-warmup', default=1, type=int,
                        help='Run `initial_warmup` * `warmup_steps` steps of threshold warmup during which threshold '
                                'stays at its `initial_threshold` value (sparsity schedule).')
        group.add_argument('--pruning-final-warmup', default=2, type=int,
                        help='Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which threshold '
                                'stays at its final_threshold value (sparsity schedule).')
        group.add_argument('--pruning-final-lambda', default=0.0, type=float,
                            help="Regularization intensity (used in conjunction with `regularization`.")
        group.add_argument('--pruning-global-topk', action='store_true', help='Global TopK on the Scores.')
        group.add_argument('--pruning-global-topk-frequency-compute', default=25, type=int,
                            help='Frequency at which we compute the TopK global threshold.')
        group.add_argument('--pruning-decay-step', default=1, type=int,
                        help='Apply pruning decay each x step.')
        group.add_argument('--pruning-decay-type', default='exp', type=str,
                        help='Pruning decay type.')
        group.add_argument('--pruning-module', default='decoder', type=str,
                        help='Pruning module.')
        group.add_argument('--ft-module', default='0', type=str,
                        help='FT module.')
        group.add_argument('--attn-separate', default=False, type=bool,
                        help='Attn separate.')
        group.add_argument('--vo-global', default=False, type=bool,
                        help='self-attn vo global sort.')
        group.add_argument('--channelpruning', default=False, type=bool,
                        help='channel pruning.')
        group.add_argument('--channelpruning-threshold', default=False, type=bool,
                        help='self-attn vo channel pruning threshold.')
        group.add_argument('--pruning-global-taylor', default=False, type=bool,
                        help='channel pruning.')
        group.add_argument('--only-mask', default=False, type=bool,
                        help='only training mask.')
        group.add_argument('--LR-weight-rank', default=16, type=int,
                        help='weight rank in LR pruning')
        group.add_argument('--LR-mask-rank', default=32, type=int,
                        help='mask rank in LR pruning')
        group.add_argument('--LR-weight-alpha', default=16, type=int,
                        help='weight alpha in LR pruning')
        group.add_argument('--LR-mask-alpha', default=16, type=int,
                        help='mask alpha in LR pruning')
        group.add_argument('--few-shot', action='store_true', default=False,
                        help='Do few-shot learning or not.')
        group.add_argument('--few-shot-train-size-per-class', default=100, type=int,
                        help='Training data size for few-shot learning')
        return self.parser

    def _add_alicemind_args(self):
        group = self.parser.add_argument_group('alicemind', 'alicemind setting')
        group.add_argument("--task_type", default=None, type=str, required=True)
        group.add_argument("--train_data_name", default=None, type=str)
        group.add_argument("--dev_data_name", default=None, type=str)
        group.add_argument("--test_data_name", default=None, type=str)   

    def _plug_check(self):
        if not self.args.train_data and not self.args.train_data_path:
            print('WARNING: No training data specified')

        self.args.cuda = torch.cuda.is_available()

        self.args.rank = int(os.getenv('RANK', '0'))
        self.args.world_size = int(os.getenv("WORLD_SIZE", '1'))

        if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
            # We are using (OpenMPI) mpirun for launching distributed data parallel processes
            local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
            local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

            # Possibly running with Slurm
            num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
            nodeid = int(os.getenv('SLURM_NODEID', '0'))

            self.args.local_rank = local_rank
            self.args.rank = nodeid*local_size + local_rank
            self.args.world_size = num_nodes*local_size

        self.args.model_parallel_size = min(self.args.model_parallel_size, self.args.world_size)
        if self.args.rank == 0:
            print('using world size: {} and model-parallel size: {} '.format(
                self.args.world_size, self.args.model_parallel_size))

        self.args.dynamic_loss_scale = False
        if self.args.loss_scale is None:
            self.args.dynamic_loss_scale = True
            if self.args.rank == 0:
                print(' > using dynamic loss scaling')

        # The args fp32_* or fp16_* meant to be active when the
        # args fp16 is set. So the default behaviour should all
        # be false.
        if not self.args.fp16:
            self.args.fp32_embedding = False
            self.args.fp32_tokentypes = False
            self.args.fp32_layernorm = False

        if self.args.ft_module != '0':
            self.args.ft_module = self.args.ft_module.split(",")
        else:
            self.args.ft_module = None

    def _alicemind_check(self):
        if self.args.dev_data_name \
         and (self.args.dev_data_name == "" or len(self.args.dev_data_name) == 0):
            self.args.dev_data_name = None
        
    def get_args(self):
        """Parse all the args."""
        self._add_model_config_args()
        self._add_fp16_config_args()
        self._add_training_args()
        self._add_evaluation_args()
        self._add_text_generate_args()
        self._add_struct_args()
        self._add_distil_args()
        self._add_palm_args()
        self._add_downstream_args()
        self._add_data_args()
        self._add_pruning_args()
        self._add_alicemind_args()
    
        # Include DeepSpeed configuration arguments
        self.parser = deepspeed.add_config_arguments(self.parser)

        self.args = self.parse_args()

        self._plug_check()

        self._alicemind_check()
        
        return self.args


    # merge config to args, config has higher priority compare to normal args, while lower than override args
    def merge_config(self, config=None):
        non_default = {
            opt.dest: getattr(self.args, opt.dest)
            for opt in self.parser._option_string_actions.values()
            if hasattr(self.args, opt.dest) and opt.default != getattr(self.args, opt.dest)
        }
        if config is None:
            return self.args
        for key, value in config.__dict__.items():
            if key in non_default.keys():
                continue
            self.args.__dict__[key] = value

        
        return self.args


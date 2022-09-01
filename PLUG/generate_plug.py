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

"""Sample Generate GPT2"""

USE_TORCH_DDP = False
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from arguments import get_args
import deepspeed
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from data_utils.wordpiece import BertTokenizer
from model import PalmModel
from model import DistributedDataParallel as DDP
from utils import print_rank_0

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def initialize_distributed(args):
    """Initialize torch.distributed."""
    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '12345')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)
    # Set the model-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

def get_model(args):
    """Build the model."""
    print_rank_0('Building PLUG model. It will take a few minutes .')
    model = PalmModel(args)
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    if args.deepspeed and args.fp16: 
        model.half()   

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)
        if args.fp32_embedding:
            model.module.model.bert.embeddings.word_embeddings.float()
            model.module.model.bert.embeddings.position_embeddings.float()
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_tokentypes:
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_layernorm:
            for name, _module in model.named_modules():
                if 'LayerNorm' in name:
                    _module.float()

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model

def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)
    if args.pre_load:
        if args.load is not None:
            from load_checkpoint import pre_load
            load_model = pre_load(mpu, args.load, args.load_iteration)
            model_dict = model.module.module.model.state_dict()
            for key in load_model:
                if key not in model_dict.keys():
                    print_rank_0('Skip key: '+key)
                else:
                    print_rank_0('Loading key: '+key)
            model.module.module.model.load_state_dict(load_model, strict=False)
        args.iteration = 0
    if not args.pre_load:
        if args.load is not None:
            #args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
            from load_checkpoint import pre_load
            load_model = pre_load(mpu, args.load, args.load_iteration)
            # model_dict = model.module.module.model.state_dict() 
            # additional FP16 Module ??
            model_dict = model.module.module.module.model.state_dict()
            for key in load_model:
                if key not in model_dict.keys():
                    print_rank_0('Skip key: '+key)
                else:
                    print_rank_0('Loading key: '+key)
            # model.module.module.model.load_state_dict(pre_load(mpu, args.load, args.load_iteration), strict=False)
            model.module.module.module.model.load_state_dict(pre_load(mpu, args.load, args.load_iteration), strict=False)
            args.iteration = 0
        else:
            args.iteration = 0

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    input_mask = torch.stack([torch.tensor([1] * len(tokens), dtype=torch.long)], 0)
    segment_ids = torch.stack([torch.tensor([0] * len(tokens), dtype=torch.long)], 0)
    tokens = tokens.view(args.batch_size, -1).contiguous()
    input_mask = input_mask.view(args.batch_size, -1).contiguous()
    segment_ids = segment_ids.view(args.batch_size, -1).contiguous()
    dec_input_ids = torch.full([args.batch_size, 1], args.cls_token_id, dtype=torch.long, device=device)
    
    input_mask[tokens == 0] = 0

    tokens = tokens.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    # Get the masks and postition ids.

    return tokens, input_mask, segment_ids, dec_input_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits

def generate_samples(model, tokenizer, args, device, length, passage):

    context_count=0
    model.eval()
    seq_length = 128
    input_length = 512
    init = True
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if mpu.get_model_parallel_rank() == 0:
                if init:
                    seq_length_tensor = torch.cuda.LongTensor([50])
                    init = False
                raw_text = passage #input("\nContext prompt (stop to exit, press enter to set output length) >>> ")
                raw_text = raw_text.replace('‘', '\'').replace('“', '\"').replace('——', '--')
                seq_length = max(1, length) 
                seq_length_tensor = torch.cuda.LongTensor([seq_length]) 
                context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
                if len(context_tokens) > input_length - 2:
                    context_tokens = context_tokens[len(context_tokens) - input_length + 2:]
                context_tokens = [tokenizer.vocab[args.cls_token]] + context_tokens + [tokenizer.vocab[args.sep_token]]
                context_length = len(context_tokens)
            else:
                context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("空"))
                context_tokens = [tokenizer.vocab[args.cls_token]] + context_tokens + [tokenizer.vocab[args.sep_token]]
                context_length = len(context_tokens)
                seq_length_tensor = torch.cuda.LongTensor([50])
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()
            pad_id = 0
            if context_length < input_length:
                context_tokens.extend([pad_id] * (input_length - context_length))

            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            torch.distributed.broadcast(seq_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            seq_length = seq_length_tensor[0].item()
            if terminate_runs == 1:
                return

            all_generate_tokens = []
            generate_tokens = []
            counter = 0
            past_key_values = None
            sequence_output = None
            vocab_size = 21128
            tokens, attention_mask, types, dec_input_ids = get_batch(context_tokens_tensor, device, args)
            while counter < seq_length:
                if counter % 128 == 0 and counter != 0:
                    generate_tokens.append(tokenizer.vocab[args.sep_token])
                    start = (context_tokens_tensor == 102).nonzero(as_tuple=True)[-1]
                    if start + len(generate_tokens) >= 512:
                        context_tokens_tensor = torch.cat([context_tokens_tensor[:start], torch.cuda.LongTensor(generate_tokens)], -1)[-512:]
                    else:
                        context_tokens_tensor[start:start+len(generate_tokens)] = torch.cuda.LongTensor(generate_tokens)
                    tokens, attention_mask, types, dec_input_ids = get_batch(context_tokens_tensor, device, args)
                    generate_tokens = []
                    sequence_output = None

                # sequence_output, _ = model.module.module.module.model.bert(tokens, types, attention_mask) 
                position_ids = torch.full([args.batch_size, 1], len(generate_tokens), dtype=torch.long, device=device)
                _, logits, sequence_output = model(tokens, types, attention_mask, dec_input_ids, attention_mask, position_ids, is_infer=True, sequence_output=sequence_output, parallel_output=False)

                partition_vocab_size = logits.size()[-1]

                logits = logits[:, -1, :]
                logits = logits / args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                prev_token = prev[0].item()
                if prev_token >= vocab_size: #or prev_token == 102:
                    prev_token = 100
                    prev[0] = 100
                if prev_token == 102 and len(all_generate_tokens) > int(max(1, length) * 0.8):
                    break
                if prev_token == 102:
                    counter += 1
                    continue
                #if prev_token == 100:
                #    counter += 1
                #    continue
                dec_input_ids = torch.cat([dec_input_ids, prev], dim=1)
                generate_tokens.append(prev_token)
                all_generate_tokens.append(prev_token)
                counter += 1

            generate_context = []
            for token in all_generate_tokens:
                if generate_context and generate_context[-1] == 100 and token == 100:
                    continue
                else:
                    generate_context.append(token)
            generate_context = "".join(tokenizer.convert_ids_to_tokens(generate_context)).replace('[UNK]', '“').replace('##','')
            return generate_context
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    while after % mpu.get_model_parallel_world_size() != 0:
        after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def setup_tokenizer(args):
    data_config = configure_data()
    data_config.set_defaults(data_set_type='BERT', transpose=False)
    tokenizer = data_config.setup_tokenizer_for_structbert(args)
    make_palm_loaders(args)
    if mpu.get_model_parallel_rank() == 0:
        args.do_train = True
        args.do_valid = True
        args.do_test = False
        before = tokenizer.num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        # Need to broadcast num_tokens and num_type_tokens.
        token_counts = torch.cuda.LongTensor([after,
                                              tokenizer.num_type_tokens,
                                              int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])
    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()
    return tokenizer, num_tokens, num_type_tokens

def make_palm_loaders(args):
    #world_size = torch.distributed.get_world_size(
    #    group=mpu.get_data_parallel_group())
    #batch_size = args.batch_size * world_size
    #we don't need multiple world_size because we don't use distributed batch sampler
    batch_size = args.batch_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size #* world_size
    seq_length = args.seq_length
    if seq_length < 0:
        seq_length = seq_length * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length * world_size

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_model_type)
    tokenizer.num_tokens = len(tokenizer.vocab)
    tokenizer.num_type_tokens = 3
    args.tokenizer = tokenizer
    args.cls_token, args.sep_token, args.mask_token = '[CLS]', '[SEP]', '[MASK]'
    args.cls_token_id = tokenizer.vocab[args.cls_token] 
    args.bos_token, args.eos_token = '[CLS]', '[SEP]'
    args.vocab_words = list(tokenizer.vocab)
    #add palm args
    args.start_length = 30
    args.tgt_length = 128
    args.full_sent_prob = 0.3
    #add structbert args
    args.environ = 'local'
    args.dataset_has_lang_id = False
    args.one_sentence = False
    args.short_seq_prob = 0
    args.ns_type = 3
    args.jieba = False
    args.do_whole_word_mask = False
    args.masked_lm_prob = 0.15
    args.do_mask_rate_range = False
    args.all_token_mlm = False
    args.predict_context_prob = 0
    args.continue_mask_prob = 0
    args.shuffle_order_prob = 0
    args.tokenizer_type = 'bert'

def get_model_tokenizer(vocab, pretrain_model_path):
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    args.model_parallel_size = 8
    args.pre_load = True
    args.palm_dataset = True
    args.num_layers = 24
    args.dec_layers = 6
    args.hidden_size = 8192
    args.num_attention_heads = 128
    args.max_position_embeddings = 2048
    args.tokenizer_type =  'BertWordPieceTokenizer'
    args.tokenizer_model_type = vocab 
    args.distributed_backend = 'nccl'
    args.fp16 = True
    args.fp32_layernorm = True
    args.checkpoint_activations = True
    args.deepspeed_activation_checkpointing = True
    args.load = pretrain_model_path
    args.load_iteration = ''
    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer, args.tokenizer_num_tokens, args.tokenizer_num_type_tokens = setup_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1
    args.top_k = 20
    args.top_p = 0.0
    args.temperature = 0.9
    return model, tokenizer, args 


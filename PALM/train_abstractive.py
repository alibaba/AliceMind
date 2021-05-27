#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import collections
import glob
import os
import random
import signal
import time

import torch
from transformers import BertTokenizer
from transformers import RobertaTokenizer

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.loss import abs_loss
from models.model_builder import AbsSummarizer
from models.predictor import build_predictor
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_abs_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_abs_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_abs(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key = lambda x: int(x.split("_")[-1][:-3]))
        print (cp_files)
        #cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        #for i, cp in enumerate(cp_files):
        for i, cp in enumerate(cp_files[::-1]):
            step = int(cp.split('.')[-2].split('_')[-1])
            test_abs(args, device_id, cp, step)
            '''
            if args.dataset == "marco":
                test_abs(args, device_id, cp, step)
                continue
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_abs(args, device_id, cp, step)
        #'''
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    #validate(args, device_id, cp, step)
                    test_abs(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    if args.encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_pth, do_lower_case=False)
        symbols = {'BOS': tokenizer.cls_token_id, 'EOS': tokenizer.sep_token_id,
                    'PAD': tokenizer.pad_token_id, 'EOQ': tokenizer.unk_token_id}
    elif args.encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[CLS]'], 'EOS': tokenizer.vocab['[SEP]'],
                    'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    elif args.encoder == 'zh_bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_pth, do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}
    model = AbsSummarizer(args, device, checkpoint, None, None)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)


    valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    encoder = args.encoder
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    #print(args)
    args.encoder = encoder 
    if args.encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_pth, do_lower_case=False)
        symbols = {'BOS': tokenizer.cls_token_id, 'EOS': tokenizer.sep_token_id,
                    'PAD': tokenizer.pad_token_id, 'EOQ': tokenizer.unk_token_id}
    elif args.encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[CLS]'], 'EOS': tokenizer.vocab['[SEP]'],
                    'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    elif args.encoder == 'zh_bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_pth, do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}

    model = AbsSummarizer(args, device, checkpoint, None, None)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, args.mode, shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def test_text_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)
    if args.encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_pth, do_lower_case=False)
        symbols = {'BOS': tokenizer.cls_token_id, 'EOS': tokenizer.sep_token_id,
                    'PAD': tokenizer.pad_token_id, 'EOQ': tokenizer.unk_token_id}
    elif args.encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[CLS]'], 'EOS': tokenizer.vocab['[SEP]'],
                    'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    elif args.encoder == 'zh_bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_pth, do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}

    model = AbsSummarizer(args, device, checkpoint, None, None)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def baseline(args, cal_lead=False, cal_oracle=False):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, 'cpu',
                                       shuffle=False, is_test=True)

    trainer = build_trainer(args, '-1', None, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)


def train_abs(args, device_id):
    if (args.world_size > 1):
        train_abs_multi(args)
    else:
        train_abs_single(args, device_id)


def train_abs_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        encoder = args.encoder
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        args.encoder = encoder
    else:
        checkpoint = None

    if (args.load_from_extractive != ''):
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        #bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    if args.encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_pth, do_lower_case=False)
        symbols = {'BOS': tokenizer.cls_token_id, 'EOS': tokenizer.sep_token_id,
                    'PAD': tokenizer.pad_token_id, 'EOQ': tokenizer.unk_token_id}
    elif args.encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[CLS]'], 'EOS': tokenizer.vocab['[SEP]'],
                    'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    elif args.encoder == 'zh_bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_pth, do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}

         
    model = AbsSummarizer(args, device, checkpoint, bert_from_extractive, None)
    if (args.sep_optim):
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)


    train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)
    if args.pretrain:
        trainer.pretrain(args.pretrain_data_pth, args.train_steps) 
    else:
        trainer.train(train_iter_fct, args.train_steps)

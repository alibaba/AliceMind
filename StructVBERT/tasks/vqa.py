# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
import logging
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from lxrt.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class WarmupOptimizer(object):
    def __init__(self, _lr_base, optimizer, _data_size, _batch_size):
        self.optimizer = optimizer
        self._step = 0
        self._lr_base = _lr_base
        self._rate = 0
        self._data_size = _data_size
        self._batch_size = _batch_size

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        if step <= int(self._data_size / self._batch_size * 1):
            r = self._lr_base * 1/4.
        elif step <= int(self._data_size / self._batch_size * 2):
            r = self._lr_base * 2/4.
        elif step <= int(self._data_size / self._batch_size * 3):
            r = self._lr_base * 3/4.
        else:
            r = self._lr_base
        return r

def adjust_learning_rate(optimizer, decay_rate):
    optimizer._lr_base *= decay_rate

class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=256,  # for large model
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)
        self._lr_decay_epoch_list = [8, 10]
        self._lr_decay_rate = 0.2

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        if args.fix_language_bert:
            assert args.patial_load
            state_dict = torch.load(args.patial_load)
            for k in state_dict.copy():
                if not k.startswith('bert.'):
                    state_dict['bert.' + k.replace('gamma', 'weight').replace('beta', 'bias')] = state_dict.pop(k)

            # fix bert parameters
            for name, param in self.model.lxrt_encoder.model.named_parameters():
                # if 'pooler' in name: # pooler not fixed
                #     continue
                if name in state_dict:
                    logger.info('fix param for: {}'.format(name))
                    param.requires_grad = False

        # GPU options
        self.model = self.model.cuda()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            logger.info("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        elif 'adam' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            optim = args.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9)
            self.optim = WarmupOptimizer(args.lr, optim, batch_per_epoch * args.batch_size, args.batch_size)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        if args.amp_type is not None:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level=args.amp_type)

        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            if 'adam' in args.optim and epoch in self._lr_decay_epoch_list:
                adjust_learning_rate(self.optim, self._lr_decay_rate)
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)
                if args.multiGPU:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                if args.amp_type is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_norm)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            logger.info(log_str)

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                if args.with_score:
                    logit = nn.Softmax(dim=1)(logit)
                score, label = logit.max(1)
                if args.with_score:
                    for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid.item()] = (ans, str(s))
                else:
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        logger.info("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            logger.info(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        # print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        logger.info('Splits in Train data: {}'.format(vqa.train_tuple.dataset.splits))
        if vqa.valid_tuple is not None:
            logger.info('Splits in Valid data: {}'.format(vqa.valid_tuple.dataset.splits))
            logger.info("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            logger.info("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)



import glob
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch

from others.logging import logger
from transformers import BertTokenizer
from transformers import RobertaTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams
import argparse
import time


class RobertaData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(args.en_pretrained_model, do_lower_case=False)

        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.tgt_bos = '<s>'
        self.tgt_eos = '</s>'
        self.tgt_sent_split = '<q>'
        self.sep_vid = self.tokenizer.sep_token_id
        self.cls_vid = self.tokenizer.cls_token_id
        self.pad_vid = self.tokenizer.pad_token_id
    def preprocess(self, src, tgt, use_bert_basic_tokenizer=False, is_test=False):


        src_subtokens = self.tokenizer.tokenize(src)
        if len(src_subtokens) > 500:
            src_subtokens = src_subtokens[:500]
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tgt_subtoken = [self.tgt_bos] + self.tokenizer.tokenize(tgt) + [self.tgt_eos]
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        
        tgt_txt = self.tokenizer.decode(tgt_subtoken_idxs).replace("<s>", "").replace("</s>", "")
        src_txt = self.tokenizer.decode(src_subtoken_idxs).replace("<s>", "").replace("</s>", "")

        return src_subtoken_idxs, tgt_subtoken_idxs, src, tgt
    def preprocess_qg(self, src, tgt, answer, use_bert_basic_tokenizer=False, is_test=False):
        text = answer + " " + self.sep_token + src
        src_subtokens = self.tokenizer.tokenize(answer) + [self.sep_token] + self.tokenizer.tokenize(src)
        if len(src_subtokens) > 508:
            src_subtokens = src_subtokens[:508]
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tgt_subtokens = self.tokenizer.tokenize(tgt)
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids([self.cls_token] + tgt_subtokens + [self.sep_token])

        tgt_txt = self.tokenizer.decode(tgt_subtoken_idxs)
        src_txt = self.tokenizer.decode(src_subtoken_idxs)

        return src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt
class ZhBertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.zh_pretrained_model, do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_sent_split = '[unused3]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, use_bert_basic_tokenizer=False, is_test=False):
        src_subtokens = self.tokenizer.tokenize(src)
        if len(src_subtokens) > 500:
            src_subtokens = src_subtokens[:500]
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tgt_subtoken = [self.tgt_bos] + self.tokenizer.tokenize(tgt) + [self.tgt_eos]
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = "".join(self.tokenizer.convert_ids_to_tokens(tgt_subtoken_idxs))
        src_txt = "".join(self.tokenizer.convert_ids_to_tokens(src_subtoken_idxs))

        return src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt




def format_to_qg(args):
    zhbert = ZhBertData(args)
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    f_reader = open(args.input_file, 'r')
    datasets = []
    for query_id, line in enumerate(f_reader):
        src, tgt = line.split("\t")
        src = "".join(src.split(" "))
        tgt = "".join(tgt.split(" "))
        src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt = zhbert.preprocess(src, tgt)
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "query_id": query_id}
        datasets.append(b_data_dict)

    f_reader.close()
    # save_file = pjoin(args.output_dir, args.task+'.'+args.corpus_type+'.0.pt')
    torch.save(datasets, args.output_file)


def format_to_squad(args):
    zhbert = RobertaData(args)
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    f_reader = open(args.input_file, 'r')
    datasets = []
    for query_id, line in enumerate(f_reader):
        src, answer = line.split("\t")
        tgt = ""
        src_subtoken_idxs, tgt_subtoken_idxs, src_txt, tgt_txt = zhbert.preprocess_qg(src, tgt, answer)
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "query_id": query_id}
        datasets.append(b_data_dict)

    f_reader.close()
    # save_file = pjoin(args.output_dir, args.task+'.'+args.corpus_type+'.0.pt')
    torch.save(datasets, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-zh_pretrained_model", default='palm_model/512-85w/', type=str)
    parser.add_argument("-en_pretrained_model", default='palm_model/roberta-base/', type=str)
    parser.add_argument("-input_file", default='', type=str)
    parser.add_argument("-output_file", default='', type=str)
    # parser.add_argument("-task", default='', type=str)
    # parser.add_argument("-corpus_type", default='train', type=str, choices=['train', 'valid', 'test'])
    args = parser.parse_args()

    format_to_squad(args)

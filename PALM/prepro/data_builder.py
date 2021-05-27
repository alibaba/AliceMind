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
from others.transformers import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams
import argparse
import time


class ZhBertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.model_pth, do_lower_case=True)

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
        torch.save(dataset, save_file)
def format_to_qg(args):
    #format_to_robert_wiki_query_generation
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train']
    raw_path = '/home/lcl193798/PreRobertaSummMaro/raw_data/dureader_zhidao_data_10k'
    save_path = '/home/lcl193798/PreRobertaSummMaro/bert_data/dureader_zhidao_data_100k'
    for corpus_type in datasets:

                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                    "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                                    'src_txt': src_txt, "tgt_txt": tgt_txt, "query_id": new_query_id}
                    datasets.append(b_data_dict)
                    break
                if len(datasets) == 10000:
                    break
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)
    parser.add_argument("")
    args = parser.parse_args()
    format_to_qg(args)

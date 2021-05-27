# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team and Alibaba-inc.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import sys
import torch
import json
from torch.utils.data import Dataset, Sampler, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import scipy.stats as sp
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat
import tokenization
from modeling import BertConfig, BertForSequenceClassificationMultiTask
from optimization import BERTAdam, Adamax

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, index):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.index = index

class FeatureDataset(Dataset):
    """Squad dataset"""
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def lengths(self):
        return [len(feature.input_ids) for feature in self.features]

class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.lengths],
            dtype=[('sent_len', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('sent_len', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

def batchify(batch):
    seq_len = max([len(feature.input_ids) for feature in batch])
    input_ids, input_mask, segment_ids, label_id, label_index = \
        list(), list(), list(), list(), list()
    for feature in batch:
        padding = [0 for _ in range(seq_len - len(feature.input_ids))]
        input_ids_ins = feature.input_ids
        input_mask_ins = feature.input_mask
        segment_ids_ins = feature.segment_ids
        input_ids_ins.extend(padding), input_mask_ins.extend(padding), segment_ids_ins.extend(padding)
        input_ids.append(torch.tensor(input_ids_ins, dtype=torch.long))
        input_mask.append(torch.tensor(input_mask_ins, dtype=torch.long))
        segment_ids.append(torch.tensor(segment_ids_ins, dtype=torch.long))
        label_id.append(torch.tensor(feature.label_id, dtype=torch.float))
        label_index.append(torch.tensor(feature.index, dtype=torch.long))
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    segment_ids = torch.stack(segment_ids, 0)
    label_id = torch.stack(label_id, 0)
    label_index = torch.stack(label_index, 0)
    return input_ids, input_mask, segment_ids, label_id, label_index

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        text_list = open(input_file).readlines()
        text_list = [json.loads(line) for line in text_list]
        return text_list

class OCNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.50k.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['label'] if set_type!='test' else 'entailment'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class CMNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['label'] if set_type!='test' else 'entailment'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AFQMCProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return ["0", "1"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['label'] if set_type!='test' else '0'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class CSLProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return ["0", "1"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = '/'.join(line['keyword'])
            text_b = line['abst']
            label = line['label'] if set_type!='test' else '0'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TNEWSProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return ["100", "101", "102", "103",
                "104", "105", "106", "107",
                "108", "109", "110", "111",
                "112", "113", "114", "115", "116"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = line['keywords']
            label = line['label'] if set_type!='test' else '100'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class IFLYTEKProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'dev.json')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), "test")
    def get_labels(self):
        return [str(i) for i in range(119)]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            label = line['label'] if set_type!='test' else '0'
            if label not in self.get_labels():
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0] if set_type!='test' else '0')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, str_code):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_{}.tsv".format(str_code))),
            "dev_{}".format(str_code))

    def get_test_examples(self, data_dir, str_code):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_{}.tsv".format(str_code))),
            "test_{}".format(str_code))

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1] if set_type!='test_matched' and set_type!='test_mismatched' else 'contradiction')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AXProcessor(DataProcessor):
    """Processor for the AX data set (GLUE version)."""

    def get_test_examples(self, data_dir, str_code):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "diagnostic.tsv")),
            "diagnostic")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode('contradiction')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QqpProcessor(DataProcessor):
    """Processor for the Qqp data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["1", "0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type!='test' and len(line) != 6:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3] if set_type!='test' else line[1])
            text_b = tokenization.convert_to_unicode(line[4] if set_type!='test' else line[2])
            label = tokenization.convert_to_unicode(line[5] if set_type!='test' else '1')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WnliProcessor(DataProcessor):
    """Processor for the Wnli data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["1", "0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3] if set_type!='test' else '1')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

 
class QnliProcessor(DataProcessor):
    """Processor for the Qnli data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3] if set_type!='test' else 'entailment')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Sts_bProcessor(DataProcessor):
    """Processor for the Sts_b data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['Regression']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[7])
            text_b = tokenization.convert_to_unicode(line[8])
            label = float(tokenization.convert_to_unicode(line[9])) if set_type!='test' else '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst_2Processor(DataProcessor):
    """Processor for the Sst_2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0] if set_type!='test' else line[1])
            label = tokenization.convert_to_unicode(line[1] if set_type!='test' else '0')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and set_type == 'test':
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1] if set_type=='test' else line[3])
            label = tokenization.convert_to_unicode(line[1] if set_type!='test' else '0')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def examples_to_features_worker(example, max_seq_length, tokenizer, label_map, index, max_index, is_training, args):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if not (is_training and args.fast_train):
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    else:
        assert len(input_ids) == len(input_mask) == len(segment_ids)

    label_id = [0] * max_index
    if len(label_map[index]) != 0:
        label_id[index] = label_map[index][example.label]
    else:
        label_id[index] = example.label

    return  InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    index=index,)    

def convert_examples_to_features(args, examples, label_lists, max_seq_length, tokenizer, index, max_index, is_training=False):
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map_lst = []
    for label_list in label_lists:
        label_map = {}
        if len(label_list)!= 1:
            for (i, label) in enumerate(label_list):
                label_map[label] = i
        label_map_lst.append(label_map)
    pool = Pool(mp.cpu_count())
    logger.info('start tokenize')
    features = pool.starmap(examples_to_features_worker, zip(examples, repeat(max_seq_length), repeat(tokenizer), repeat(label_map_lst), repeat(index), repeat(max_index), repeat(is_training), repeat(args)))
    pool.close()
    pool.join()

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def matthew_corr(tp, tn, fp, fn):
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denominator == 0:
        return 0
    else:
        return (tp*tn-fp*fn)/denominator
    

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--optimizer",
                        default='adam',
                        type=str,
                        help="Optimizer type.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--pretrain_model",
                        default=None,
                        type=str,
                        help="Used to ontinue training.")
    parser.add_argument("--core_encoder",
                        default='bert',
                        type=str,
                        help="core encoder, support 'bert' or 'lstm'.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--gradual_unfreezing",
                        default=False,
                        action='store_true',
                        help="layerwise unfreezing gradient or not")
    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        help="debug mode")
    parser.add_argument("--add_dialog_token",
                        default=False,
                        action='store_true',
                        help="add [SLR][BYR] token when process dialog data")
    parser.add_argument("--sequential",
                        default=False,
                        action='store_true',
                        help='Different task will be sequential trained if True')
    parser.add_argument("--lr_decay_factor",
                        default=1,
                        type=float,
                        help='lr decay factor over all layers')
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help='dropout during downstream task')
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--detach_index",
                        default=-1,
                        type=int,
                        help="fix layers of transformer during finetune")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument('--save_model',
                        default=False,
                        action='store_true',
                        help='save checkpoint or not')
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fast_train',
                        default=False,
                        action='store_true',
                        help='sort examples by length')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--num_workers", default=16, type=int, help="data loader workers")
    parser.add_argument("--amp_type", default=None, type=str, help="whether to use mix precision, must in [O0, O1, O2, O3]")
    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "rte": QnliProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "qnliv2": QnliProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst_2Processor,
        "sts-b": Sts_bProcessor,
        "wnli": WnliProcessor,
        "wnliv2": WnliProcessor,
        "ocnli_public": OCNLIProcessor,
        "afqmc_public": AFQMCProcessor,
        "tnews_public": TNEWSProcessor,
        "iflytek_public": IFLYTEKProcessor,
        "cmnli_public": CMNLIProcessor,
        "csl_public": CSLProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config.attention_probs_dropout_prob = args.dropout
    bert_config.hidden_dropout_prob = args.dropout

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name_list = [task.lower() for task in args.task_name.split(',')]

    for task_name in task_name_list:
        if task_name not in processors and task_name != 'none':
            raise ValueError("Task not found: %s" % (task_name))

    processor_list = [processors[task_name]() if task_name != 'none' else None for task_name in task_name_list]
    label_list = [processor.get_labels() if processor is not None else ['None'] for processor in processor_list]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = []
        for task_name, processor in zip(args.task_name.split(','), processor_list):
            data_dir = os.path.join(args.data_dir, task_name)
            if task_name.lower() == 'none':
                train_examples.append([])
                continue
            if not args.debug:
                if not args.add_dialog_token:
                    train_examples.append(processor.get_train_examples(data_dir))
                else:
                    train_examples.append(processor.get_train_examples(data_dir, add_token=True))
            else:
                train_examples.append(processor.get_train_examples(data_dir)[:args.train_batch_size * 5])
        num_train_steps = int(
            sum([len(exs) for exs in train_examples]) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassificationMultiTask(bert_config, label_list, args.core_encoder)
    if args.init_checkpoint is not None:
        try:
            model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
        except:
            new_state_dict = {}
            state_dict = torch.load(args.init_checkpoint, map_location='cuda')
            for key in state_dict:
                if key.startswith('bert.'):
                    new_state_dict[key[5:]] = state_dict[key]
                elif key.startswith('module.bert.'):
                    new_state_dict[key[12:]] = state_dict[key]
                else:
                    pass
            model.bert.load_state_dict(new_state_dict)       
    if args.pretrain_model is not None:
        try:
            model.load_state_dict(torch.load(args.pretrain_model, map_location='cuda'))
        except:
            new_state_dict = dict()
            state_dict = torch.load(args.pretrain_model, map_location='cuda')
            for item in state_dict:
                if item.lower().startswith('module'):
                    new_item = item[7:]
                else:
                    new_item = item
                new_state_dict[new_item] = state_dict[item]
            model_dict = model.state_dict()
            same_state_dict = {k: v for k, v in new_state_dict.items() if v.size() == model_dict[k].size()}
            model_dict.update(same_state_dict)
            model.load_state_dict(model_dict)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    param_lst = []
    param_lst.append([p for n, p in param_optimizer[:21]])
    start = 21
    for i in range(bert_config.num_hidden_layers - 2):
        param_lst.append([p for n, p in param_optimizer[start:start+16]])
        start += 16
    param_lst.append([p for n, p in param_optimizer[start:]])
    optimizer_grouped_parameters = []
    init_lr = args.learning_rate
    for i in range(bert_config.num_hidden_layers):
        optimizer_grouped_parameters.append({'params': param_lst[i], 'lr':init_lr})
        init_lr /= args.lr_decay_factor ** (1.0 / bert_config.num_hidden_layers)
    if args.optimizer.lower() == 'adam':
        optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    elif args.optimizer.lower() == 'adamax':
        optimizer = Adamax(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    else:
        raise ValueError("Must be adam or adamax, but got{}".format(args.optimizer))
    if args.amp_type is not None:
        try:
            import apex.amp as amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_type)
        except:
            logger.info('Install apex first if you want to use mix precition.')

    global_step = 0
    if args.do_train:
        logger.info("***** Process training data *****")
        train_features = []
        max_index = len(train_examples)
        for index, train_examples_spec in enumerate(train_examples):
            train_features.extend(convert_examples_to_features(
                args, train_examples_spec, label_list, args.max_seq_length, tokenizer, index, max_index, is_training=True))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Num tasks = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        if not args.fast_train:
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
            all_label_index = torch.tensor([f.index for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_index)
            if args.local_rank == -1:
                if args.sequential:
                    train_sampler = SequentialSampler(train_data)
                else:
                    train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        else:
            train_data = FeatureDataset(train_features)
            train_sampler = SortedBatchSampler(train_data.lengths(),
                                           args.train_batch_size,
                                           shuffle=not args.sequential)
            train_dataloader = DataLoader(train_data,
                                       batch_size=args.train_batch_size,
                                       sampler = train_sampler,
                                       num_workers=args.num_workers,
                                       collate_fn=batchify,
                                       drop_last=len(train_data) % args.train_batch_size == 1)

    if args.do_eval:
        logger.info("***** Process dev data *****")
        eval_examples = []
        for task_name, processor in zip(args.task_name.split(','), processor_list):
            data_dir = os.path.join(args.data_dir, task_name)
            if task_name.lower() == 'none':
                eval_examples.append([])
                continue
            if task_name.lower() == 'mnli':
                eval_examples.append(processor.get_dev_examples(data_dir, 'matched') + processor.get_dev_examples(data_dir, 'mismatched'))
            else:
                if not args.add_dialog_token:
                    eval_examples.append(processor.get_dev_examples(data_dir))
                else:
                    eval_examples.append(processor.get_dev_examples(data_dir, add_token=True))
        max_eval_index = len(eval_examples)
        eval_features = []
        for index, eval_examples_spec in enumerate(eval_examples):
            eval_features.extend(convert_examples_to_features(
                args, eval_examples_spec, label_list, args.max_seq_length, tokenizer, index, max_eval_index))
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Num tasks = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        all_label_index = torch.tensor([f.index for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_test:
        logger.info("***** Process test data *****")
        test_examples = []
        for task_name, processor in zip(args.task_name.split(','), processor_list):
            data_dir = os.path.join(args.data_dir, task_name)
            if task_name.lower() == 'none':
                test_examples.append([])
                continue
            if task_name.lower() == 'mnli':
                diagnostic_dir = os.path.join(args.data_dir, 'diagnostic')
                test_examples.append(processor.get_test_examples(data_dir, 'matched') + \
                    processor.get_test_examples(data_dir, 'mismatched') + AXProcessor().get_test_examples(diagnostic_dir, 'diagnostic'))
            else:
                if not args.add_dialog_token:
                    test_examples.append(processor.get_test_examples(data_dir))
                else:
                    test_examples.append(processor.get_test_examples(data_dir, add_token=True))
        max_test_index = len(test_examples)
        test_features = []
        for index, test_examples_spec in enumerate(test_examples):
            test_features.extend(convert_examples_to_features(
                args, test_examples_spec, label_list, args.max_seq_length, tokenizer, index, max_test_index))
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Num tasks = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_index = torch.tensor([f.index for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_index)
        if args.local_rank == -1:
            test_sampler = SequentialSampler(test_data)
        else:
            test_sampler = DistributedSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        for epoch_id in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_index = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, label_index, epoch_id=epoch_id if args.gradual_unfreezing else -1)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.amp_type is not None:
                    try:
                        import apex.amp as amp
                        with amp.scale_loss(loss, optimizer) as loss:
                            loss.backward()
                    except:
                        logger.info('Install apex first if you want to use mix precition.')
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
            if args.do_eval:
                model.eval()
                eval_loss, eval_accuracy = 0, [0] * max_eval_index
                eval_corrcoef = [([],[])] * max_eval_index
                matthew_cnt = [(0, 0, 0, 0)] * max_eval_index #TP, TF, FP, FN
                nb_eval_steps, nb_eval_examples = 0, [0] * max_eval_index

                eval_logits_files = [open(os.path.join(args.output_dir, task_name+".eval_logits_EP{}".format(epoch_id)), 'w') \
                                        for task_name in task_name_list]

                for input_ids, input_mask, segment_ids, label_ids, label_index in eval_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    label_index = label_index.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, label_index)

                    logits = [logit.detach().cpu().numpy() for logit in logits]
                    label_ids = label_ids.to('cpu').numpy()
                    label_index = label_index.to('cpu').numpy()
                    tmp_eval_accuracy = []
                    tmp_eval_corrcoef = []
                    for index, (logit, label_id) in enumerate(zip(logits, np.split(label_ids, max_eval_index, axis=1))):
                        for is_match, single_label, ins_logit in zip((index == label_index).tolist(), label_id.tolist(), logit.tolist()):
                            if is_match:
                                eval_logits_files[index].write(str(single_label)+'\t'+str(ins_logit)+'\n')
                        if len(label_list[index]) != 1:
                            outputs = np.argmax(logit, axis=1)
                            tmp_eval_accuracy.append(np.sum((outputs.flatten() == label_id.flatten()) * (index == label_index)))
                            tmp_eval_corrcoef.append(([],[]))
                            if len(label_list[index]) == 2 and task_name_list[index] == 'cola':
                                 TP = np.sum((outputs.flatten() == label_id.flatten()) * (outputs.flatten() == 1) * (index == label_index))
                                 TN = np.sum((outputs.flatten() == label_id.flatten()) * (outputs.flatten() == 0) * (index == label_index))
                                 FP = np.sum((outputs.flatten() == 1) * (outputs.flatten() != label_id.flatten()) * (index == label_index))
                                 FN = np.sum((outputs.flatten() == 0) * (outputs.flatten() != label_id.flatten()) * (index == label_index))
                                 matthew_cnt[index] = (matthew_cnt[index][0]+TP, matthew_cnt[index][1]+TN, \
                                                       matthew_cnt[index][2]+FP, matthew_cnt[index][3]+FN)
                        else:
                            tmp_eval_accuracy.append(0)
                            batch_outputs, batch_label = [],[]
                            for single_label, single_logit, flag in zip(logit.flatten(), label_id.flatten(), index == label_index):
                                if flag:
                                    batch_outputs.append(single_logit)
                                    batch_label.append(single_label)
                            tmp_eval_corrcoef.append((batch_outputs, batch_label))
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy = [a + b for a, b in zip(eval_accuracy, tmp_eval_accuracy)]
                    eval_corrcoef = [(a[0]+b[0], a[1]+b[1]) for a, b, lst in zip(eval_corrcoef, tmp_eval_corrcoef, label_list)]
                    for index in range(max_eval_index):
                        nb_eval_examples[index] += np.sum(index == label_index)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = [a/b for a, b in zip(eval_accuracy,nb_eval_examples)]
                eval_corrcoef = [sp.pearsonr(a[0], a[1])[0] if a!=([], []) else 0 for a in eval_corrcoef]
                matthew_cnt = [matthew_corr(a[0], a[1], a[2], a[3]) for a in matthew_cnt] 
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          'eval_corrcoef': eval_corrcoef,
                          'eval_matthew': matthew_cnt,
                          'global_step': global_step,
                          'loss': tr_loss/(nb_tr_steps+1e-6)}
                output_eval_file = os.path.join(args.output_dir, args.task_name+".eval_results_EP{}".format(epoch_id))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("%s:  %s:  %s = %s", args.init_checkpoint if args.init_checkpoint is not None else args.pretrain_model, args.task_name, key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
            if args.do_test:
                #####################run prediction####################
                instance_idx = 0
                output_test_files = [open(os.path.join(args.output_dir, task_name+".test_results_EP{}".format(epoch_id)), 'w') \
                                        for task_name in task_name_list]
                output_logits_files = [open(os.path.join(args.output_dir, task_name.upper()+".test_results_EP{}".format(epoch_id)), 'w') \
                                        for task_name in task_name_list]
                headers = ['id', 'label']
                f_csvs = [csv.writer(handle, delimiter="\t") for handle in output_test_files]
                logits_csvs = [csv.writer(handle, delimiter="\t") for handle in output_logits_files]
                test_result, test_logits_result = [], []
                for f_csv, logits_csv in zip(f_csvs, logits_csvs):
                    f_csv.writerow(headers)
                    logits_csv.writerow(['id', 'label', 'logits'])
                    test_result.append([])
                    test_logits_result.append([])
                for input_ids, input_mask, segment_ids, label_index in tqdm(test_dataloader):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_index = label_index.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None, labels_index=label_index)

                    logits = [logit.detach().cpu().numpy() for logit in logits]
                    label_index = label_index.to('cpu').numpy()
                    
                    for index, logit in enumerate(logits):
                        if len(label_list[index]) != 1:
                            outputs = np.argmax(logit, axis=1).tolist()
                        else:
                            outputs = logit.tolist()
                        for is_match, predict_ins, labels, ins_logit in zip((index == label_index).tolist(), outputs, label_index.tolist(), logit.tolist()):
                            if is_match:
                                if len(label_list[index]) != 1:
                                    test_result[index].append((instance_idx, label_list[labels][predict_ins]))
                                    test_logits_result[index].append((instance_idx, label_list[labels][predict_ins], ins_logit))
                                else:
                                    test_result[index].append((instance_idx, predict_ins[0]))
                                    test_logits_result[index].append((instance_idx, predict_ins[0], logit))
                                instance_idx += 1
                for i in range(len(test_result)):
                    f_csvs[i].writerows(test_result[i])
                    logits_csvs[i].writerows(test_logits_result[i])
            if args.save_model: 
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(args.output_dir, args.task_name+".model.ep{}".format(epoch_id)))
                logger.info("  Save model of Epoch %d", epoch_id)
if __name__ == "__main__":
    main()

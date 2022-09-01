import os
import json
import torch
import random
import collections
import logging
import h5py
import copy


import torch.distributed as dist
from torch.utils.data import Dataset, Subset
from multiprocessing import Pool
from itertools import repeat
import multiprocessing as mp
from .tokenization_plug import BertTokenizer, printable_text
from sofa.utils import mpu, data_utils, print_rank_0

# logger
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class TaskTypeName(object):
  SINGLE_CLASSIFICATION = 'single_classification'
  PAIR_CLASSIFICATION = 'pair_classification'
  SEQUENCE_LABELING = 'sequence_labeling'
  SEQUENCE_LABELING_CRF = 'sequence_labeling_crf'
  CHINESE_MRC_SD = 'chinese_mrc_sd'
  GENERATION = 'generation'
  STRUCTBERT_EMBEDDING = "structbert_embedding"
  MULTILABEL_CLASSIFICATION = "multilabel_classification"
  SENTENCE_SIMILARITY = "sentence_sim"
  SENTENCE_CLUSTER = "sentence_cluster"
  MULTICHOICE_MRC = "multichoice_mrc"
  SINGLE_REGRESSION = "single_regression"
  PAIR_REGRESSION = "pair_regression"
  KBQA = "kb_question_answering"   # knowledge base question answering

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
    
    def get_label_id(self):
        return self.label_id
    
    def get_example_item(self, item):
        return
    
    def _create_examples(self, lines, set_type):
        """Generate examples."""
        raise NotImplementedError()

    
    @classmethod
    def _read_json(self, input_file):
        text_list = open(input_file).readlines()
        text_list = [json.loads(line) for line in text_list]
        return text_list
    
    @classmethod
    def _read_txt(self, input_file):
        text_list = open(input_file).readlines()
        return text_list
    
    def train_dev_split(self, train_data, ratio=0.8):
        train_examples = []
        dev_examples = []
        lines = self._read_txt(train_data)
        labels = set()
        for (i, line) in enumerate(lines):
            label = None
            label_id = self.get_label_id()
            if label_id is not None:
                if len(line) == 0 or (label_id >= 0 and len(line) <= label_id):
                    continue  
                label, item = self.get_example_item(i, line, 'train')
            # make sure the train sets get all labels
            if label is not None and label not in self.get_labels():
                labels.append(label)
                train_examples.append(item)
            else:
                if random.random() < ratio:
                    train_examples.append(item)
                else:
                    item.guid = "%s-%s" % ("dev", i)
                    dev_examples.append(item)
                
        return train_examples, dev_examples, labels

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
        print("datq_dir is",data_dir)
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

class SingleSentenceClassificaitonProcessor(DataProcessor):

    def  __init__(self):
        self.labels = []
        self.text_a_id = 0
        self.label_id = 1

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'test.txt')), "test")
    def get_labels(self):
        return self.labels  # ["entailment", "neutral", "contradiction"]
     
    def get_example_item(self, index, item, set_type):
        guid = "%s-%s" % (set_type, index)
        sp = item.strip().split('\t')
        text_a = sp[0]
        label = sp[1] if set_type!='test' else self.labels[0]
        return label, InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            label, item = self.get_example_item(i, lines[i], set_type)
            if label not in self.get_labels():
                self.labels.append(label)
                # continue
            examples.append(item)
        return examples 

class SingleSentenceMutlipleClassificaitonProcessor(DataProcessor):

    def  __init__(self):
        self.labels = []
        self.text_a_id = 0
        self.label_id = 1
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'test.txt')), "test")
    def get_labels(self):
        return self.labels  # ["entailment", "neutral", "contradiction"]
     
    def get_example_item(self, index, item, set_type):
        guid = "%s-%s" % (set_type, index)
        sp = item.strip().split('\t')
        text_a = sp[0]
        label = sp[1].split() if set_type!='test' else [self.labels[0]]
        return label, InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            label, item = self.get_example_item(i, lines[i], set_type)

            if label not in self.get_labels():
                self.labels.append(label)
                # continue
            examples.append(item)
        return examples 

class MultipleSentenceClassificaitonProcessor(DataProcessor):

    def  __init__(self):
        self.labels = []
        self.text_a_id = 0
        self.text_b_id = 1
        self.label_id = 2
    def get_label_id(self):
        return self.label_id
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'test.txt')), "test")
    def get_labels(self):
        return self.labels  # ["entailment", "neutral", "contradiction"]
    def get_example_item(self, index, item, set_type):
        guid = "%s-%s" % (set_type, index)
        sp = item.strip().split('\t')
        text_a = sp[0]
        text_b = sp[1]
        label = sp[2] if set_type!='test' else self.labels[0]
        return label, InputExample(guid=guid, text_a=text_a, text_b=None, label=label)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            label, item = self.get_example_item(i, lines[i], set_type)

            if label not in self.get_labels():
                self.labels.append(label)
                # continue
            examples.append(item)
        return examples 

class MultipleSentenceMultipleClassificaitonProcessor(DataProcessor):

    def  __init__(self):
        self.labels = []
        self.text_a_id = 0
        self.text_b_id = 1
        self.label_id = 2
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'test.txt')), "test")
    def get_labels(self):
        return self.labels  # ["entailment", "neutral", "contradiction"]
    
    def get_example_item(self, index, item, set_type):
        guid = "%s-%s" % (set_type, index)
        sp = item.strip().split('\t')
        text_a = sp[0]
        text_b = sp[1]
        label = sp[2].split(' ') if set_type!='test' else [self.labels[0]]
        return label, InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            label, item = self.get_example_item(i, lines[i], set_type)
            if label not in self.get_labels():
                self.labels.append(label)
                # continue
            examples.append(item)
        return examples 

class SequenceLabelingSentenceClassificaitonProcessor(DataProcessor):

    def  __init__(self):
        self.labels = []

    def _read_txt(self, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    if len(labels) == 0 and len(words) == 0:
                        continue
                    l = ' '.join(labels)
                    w = ' '.join(words)                    
                    lines.append([l, w]) 
                    words = []
                    labels = []                
                    continue
                word = contends.split('\t')[0]
                label = contends.split('\t')[-1]
                words.append(word)
                labels.append(label)
                if label not in self.get_labels():
                    self.labels.append(label)

            if len(labels) == 0 and len(words) == 0:
                pass
            else:
                l = ' '.join(labels)
                w = ' '.join(words)                    
                lines.append([l, w]) 
            return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'test.txt')), "test")
    def get_labels(self):
        return self.labels  # ["entailment", "neutral", "contradiction"]

    def get_example_item(self, index, item, set_type):
        guid = "%s-%s" % (set_type, index)
        text = item[1]
        label = item[0]
        return label, InputExample(guid=guid, text_a=text, text_b=None, label=label)
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            label, item = self.get_example_item(i, line, set_type)
            examples.append(item)
        return examples

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
        feat = self.features[index]
        feat_dict = {'input_ids': feat.input_ids,
                     'input_mask': feat.input_mask,
                     'segment_ids': feat.segment_ids,
                     'label_id': feat.label_id,
                     'index': feat.index
                    }
        return feat_dict
        #return self.features[index]

    def lengths(self):
        return [len(feature.input_ids) for feature in self.features]

class DataConfig:
    
    def __init__(self, tokenizer, defaults={}):
        super(DataConfig, self).__init__()
        self.defaults = defaults
        self.tokenizer = tokenizer

    def apply(self, args, train=None, dev=None, test=None):
        if torch.distributed.get_rank() == 0:
            print('configuring data')
        self.apply_defaults(args)
        if args.struct_bert_dataset:
            return make_structbert_loaders(args, self.tokenizer)
        elif args.palm_dataset:
            return make_palm_loaders(args, self.tokenizer)
        elif args.downstream_dataset:
            return make_downstream_loaders(args, self.tokenizer, train, dev, test)
        else:
            return make_loaders(args)

    def setup_tokenizer_for_structbert(self, args):
        tokenizer = self.tokenizer
        tokenizer.num_tokens = len(tokenizer.vocab)
        tokenizer.num_type_tokens = 3
        return tokenizer

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            self.defaults[k] = v

    def apply_defaults(self, args):
        for k, v in self.defaults.items():
            k = k.replace('-', '_')
            if not hasattr(args, k):
                setattr(args, k, v)

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        random.seed(self.seed + id * 1000000)

class HDF5Dataset(Dataset):
    """HDF5 dataset"""
    def __init__(self, args, input_file, tokenizer, vocab_words, iter_per_epoch, is_training=True):
        self.oss_file = input_file #[dist.get_rank()]
        self.input_file = self.oss_file.split('/')[-1]+str(dist.get_rank()) if args.environ == 'jiuding' else self.oss_file
        self.input_json_file = self.input_file+'.json' if args.environ == 'jiuding' else self.oss_file.replace('token.', 'token.json') 

        self.is_training = is_training
        self.args = args
        self.sent_to_doc = json.load(open(self.input_json_file,'r'))['dict'] 
        self.total_len = len(self.sent_to_doc) 
        self.vocab_words = vocab_words
        self.tokenizer = tokenizer
        self.iter_per_epoch = iter_per_epoch

    def __len__(self):
        return self.iter_per_epoch

    def remove_local_file(self):
        os.remove(self.input_file)
        os.remove(self.input_json_file)
        logger.info("Remove local token file & json file!")

    def __getitem__(self, index):
        with h5py.File(self.input_file, 'r') as all_documents:
            document_index = str(self.sent_to_doc[random.randint(0, self.total_len - 1)])
            instance = create_instances_from_document(
                self.sent_to_doc, self.args, document_index, all_documents, self.vocab_words, random, self.tokenizer)
            feature = convert_instance_to_feature(self.args, instance, self.tokenizer)
        return feature

class PalmHDF5Dataset(Dataset):
    """HDF5 dataset"""
    def __init__(self, args, input_file, tokenizer, vocab_words, iter_per_epoch, is_training=True):
        self.oss_file = input_file #[dist.get_rank()]
        self.input_file = self.oss_file.split('/')[-1]+str(dist.get_rank()) if args.environ == 'jiuding' else self.oss_file
        self.input_json_file = self.input_file+'.json' if args.environ == 'jiuding' else self.oss_file.replace('token.', 'token.json') 

        self.is_training = is_training
        self.args = args
        self.sent_to_doc = json.load(open(self.input_json_file,'r'))['dict'] 
        self.total_len = len(self.sent_to_doc) 
        self.vocab_words = vocab_words
        self.tokenizer = tokenizer
        self.iter_per_epoch = iter_per_epoch

    def __len__(self):
        return self.iter_per_epoch

    def remove_local_file(self):
        os.remove(self.input_file)
        os.remove(self.input_json_file)
        logger.info("Remove local token file & json file!")

    def __getitem__(self, index):
        with h5py.File(self.input_file, 'r') as all_documents:
            document_index = str(self.sent_to_doc[random.randint(0, self.total_len - 1)])
            instance, lenth = create_instances_from_document(
                self.args, document_index, all_documents, self.vocab_words, random, self.tokenizer, index)
            feature = convert_instance_to_feature(self.args, instance, self.tokenizer, lenth)
        return feature

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                   next_sent_label):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.next_sent_label = next_sent_label
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "next_sent_label: %s\n" % self.next_sent_label
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def get_train_val_test_data_clean(args, tokenizer, train, dev, test):
    (train_data, val_data, test_data) = (None, None, None)
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data(tokenizer)
        data_config.set_defaults(data_set_type='BERT', transpose=False)
        (train_data, val_data, test_data), _ = data_config.apply(args, train, dev, test)
    return train_data, val_data, test_data

def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='BERT', transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(args)
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

    return train_data, val_data, test_data, num_tokens, num_type_tokens

def configure_data(tokenizer):
    
    """add cmdline flags for configuring datasets"""
    # These are options that are used by data_utils, but are either
    # deprecated or not meant to be exposed to the command line user.
    # These options are intneded to be set in code by specific scripts.
    defaults = {
        'world_size': 1,
        'rank': -1,
        'persist_state': 0,
        'lazy': False,
        'transpose': False,
        'data_set_type': 'supervised',
        'seq_length': 256,
        'eval_seq_length': 256,
        'samples_per_shard': 100
    }

    return DataConfig(tokenizer, defaults=defaults)

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
    # logger.info('start tokenize')
    if args.task_type == TaskTypeName.SEQUENCE_LABELING or args.task_type == TaskTypeName.SEQUENCE_LABELING_CRF:
        worker = ner_examples_to_features_worker
    # elif args.task_type == TaskTypeName.MULTICHOICE_MRC:
    #     worker = mrc_examples_to_features_worker
    else:
        worker = examples_to_features_worker
    features = pool.starmap(worker, zip(examples, repeat(max_seq_length), repeat(tokenizer), repeat(label_map_lst), repeat(index), repeat(max_index), repeat(is_training), repeat(args)))
    pool.close()
    pool.join()
    #features = [examples_to_features_worker(x, max_seq_length, tokenizer, label_map_lst, index, max_index, is_training, args) for x in examples]
    return features

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

def label_subword_BIO(index, label, **kwargs):
    '''
        index: int, the index of subword in the word
        label: str, the label of word
        Example:
                word A has two subwords A1 and ##A2
                label B-ORG

                A1      ##A2
        index   0       1
        return  B-ORG   I-ORG
    '''
    if index == 0:
        return label
    elif label.startswith("B-"):
        return "I-" + label[2:]
    else:
        return label

def ner_examples_to_features_worker(example, max_seq_length, tokenizer, label_map, index, max_index, is_training, args):
    
    wordlist = example.text_a.split(' ')
    labellist = example.label.split(' ')

    tokens = []
    labels = []
    segment_ids = []
    tokens.append("[CLS]")
    labels.append("O")
    segment_ids.append(0)
    for widx, word in enumerate(wordlist):
        segmented = tokenizer.tokenize(word)
        for sidx, token in enumerate(segmented):
            tokens.append(token)
            labels.append(label_subword_BIO(index = sidx, label = labellist[widx]))
            segment_ids.append(0)
    tokens.append("[SEP]")
    labels.append("O")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = [label_map[label] for label in labels]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1] + [input_ids[-1]]
        label_ids = label_ids[:max_seq_length - 1] + [label_ids[-1]]
        input_mask = input_mask[:max_seq_length - 1] + [input_mask[-1]]
        segment_ids = segment_ids[:max_seq_length - 1] + [segment_ids[-1]]

    # Zero-pad up to the sequence length.
    if not (is_training and args.fast_train):
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
    else:
        assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids)

    return  InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_ids,
                    index=index,)

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

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def make_data_loader(dataset, batch_size, args):
    
    shuffle = args.shuffle
    if shuffle:
        #if not args.struct_bert_dataset and not args.palm_dataset:
        #    sampler = data_utils.samplers.RandomSampler(dataset, replacement=True, num_samples=batch_size*args.train_iters)
        #else:
        if 1:
            sampler = data_utils.samplers.RandomSampler(dataset, replacement=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    distributed = world_size > 1
    drop_last = distributed

    if not args.struct_bert_dataset and not args.palm_dataset and not args.image_dataset:
        if distributed:
            batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler,
                                                                        batch_size,
                                                                        shuffle, #if not shuffle, than don't drop_last
                                                                        rank,
                                                                        world_size)
        else:
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size,
                                                          shuffle) #if not shuffle, than don't drop_last
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
    else:
        _worker_init_fn = WorkerInitObj(args.seed + torch.distributed.get_rank())
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  collate_fn=batchify if args.struct_bert_dataset else PalmBatchify,
                                                  worker_init_fn=_worker_init_fn)
    return data_loader

def make_tfrecord_loaders(args):
    """Load train/val/test dataset from shuffled TFRecords"""

    #import data_utils.tf_dl
    data_set_args = {'batch_size': args.batch_size,
                     'max_seq_len': args.seq_length,
                     'max_preds_per_seq': args.max_preds_per_seq,
                     'train': True,
                     'num_workers': max(args.num_workers, 1),
                     'seed': args.seed + args.rank + 1,
                     'threaded_dl': args.num_workers > 0
                     }
    train = data_utils.tf_dl.TFRecordDataLoader(args.train_data,
                                                **data_set_args)
    data_set_args['train'] = False
    if args.eval_seq_length is not None:
        data_set_args['max_seq_len'] = args.eval_seq_length
    if args.eval_max_preds_per_seq is not None:
        data_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    valid = None
    if args.valid_data is not None:
        valid = data_utils.tf_dl.TFRecordDataLoader(args.valid_data,
                                                    **data_set_args)
    test = None
    if args.test_data is not None:
        test = data_utils.tf_dl.TFRecordDataLoader(args.test_data,
                                                   **data_set_args)
    tokenizer = data_utils.make_tokenizer(args.tokenizer_type,
                                          train,
                                          args.tokenizer_path,
                                          args.vocab_size,
                                          args.tokenizer_model_type,
                                          cache_dir=args.cache_dir)

    return (train, valid, test), tokenizer

def make_downstream_loaders(args, tokenizer, train, valid, test):
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    batch_size = args.batch_size * world_size
    eval_batch_size = args.eval_batch_size * world_size 
    tokenizer.num_tokens = len(tokenizer.vocab)
    tokenizer.num_type_tokens = 3
    args.do_train = True
    args.do_valid = True
    args.do_test = True
    train = make_data_loader(train, batch_size, args)
    valid = make_data_loader(valid, eval_batch_size, args)
    shuffle = args.shuffle
    args.shuffle = False
    test = make_data_loader(test, eval_batch_size, args)
    args.shuffle = shuffle
    return (train, valid, test), tokenizer

def make_structbert_loaders(args, tokenizer):
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
        seq_length = seq_length# * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length# * world_size
    split = get_split(args)

    tokenizer.num_tokens = len(tokenizer.vocab)
    tokenizer.num_type_tokens = 3
    args.tokenizer = tokenizer
    args.cls_token, args.sep_token, args.mask_token = '[CLS]', '[SEP]', '[MASK]'
    args.vocab_words = list(tokenizer.vocab)

    #add structbert args
    #args.environ = 'local'
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

    args.do_train = True
    args.do_valid = True
    args.do_test = False
    data_parallel_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    train = HDF5Dataset(args,
                        args.sub_train_lst[data_parallel_rank],
                        args.tokenizer,
                        args.vocab_words,
                        args.train_iters * args.gradient_accumulation_steps * args.batch_size // args.num_epochs,
                        is_training=True)  
    valid = Subset(train, list(range(args.eval_iters * eval_batch_size)))
    train = make_data_loader(train, batch_size, args) 
    valid = make_data_loader(valid, eval_batch_size, args)
    return (train, valid, None), tokenizer 

def make_loaders(args):
    """makes training/val/test"""

    if args.use_tfrecords:
        return make_tfrecord_loaders(args)
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    seq_length = args.seq_length
    if seq_length < 0:
        seq_length = seq_length * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length * world_size
    split = get_split(args)
    data_set_args = {
        'path': args.train_data,
        'seq_length': seq_length,
        'lazy': args.lazy_loader,
        'delim': args.delim,
        'text_key': args.text_key,
        'label_key': 'label',
        'non_binary_cols': None,
        'ds_type': args.data_set_type,
        'split': split,
        'loose': args.loose_json,
        'tokenizer_type': args.tokenizer_type,
        'tokenizer_model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir,
        'max_preds_per_seq': args.max_preds_per_seq,
        'presplit_sentences': args.presplit_sentences}

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]
    # if optional eval args were set then replace their
    # equivalent values in the arg dict
    if eval_seq_length:
        eval_set_args['seq_length'] = eval_seq_length
    if args.eval_max_preds_per_seq:
        eval_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    if args.eval_text_key is not None:
        eval_set_args['text_key'] = args.eval_text_key

    # make datasets splits and tokenizer
    train = None
    valid = None
    test = None

    if args.train_data is not None:
        print(data_set_args)
        train, tokenizer = data_utils.make_dataset(**data_set_args)
        if data_utils.should_split(split):
            train, valid, test = train
        eval_set_args['tokenizer'] = tokenizer

    # make training and val dataset if necessary
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid, tokenizer = data_utils.make_dataset(**eval_set_args)
        eval_set_args['tokenizer'] = tokenizer
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test, tokenizer = data_utils.make_dataset(**eval_set_args)

    # wrap datasets with data loader
    if train is not None and args.batch_size > 0:
        train = make_data_loader(train, batch_size, args)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
    if valid is not None:
        valid = make_data_loader(valid, eval_batch_size, args)
        args.do_valid = True
    else:
        args.do_valid = False
    if test is not None:
        test = make_data_loader(test, eval_batch_size, args)
        args.do_test = True
    else:
        args.do_test = False

    return (train, valid, test), tokenizer

def get_split(args):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1-split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s/final_sum for s in splits]

def make_palm_loaders(args, tokenizer):
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
        seq_length = seq_length# * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length# * world_size
    split = get_split(args)

    tokenizer.num_tokens = len(tokenizer.vocab)
    tokenizer.num_type_tokens = 3
    args.tokenizer = tokenizer
    args.cls_token, args.sep_token, args.mask_token = '[CLS]', '[SEP]', '[MASK]'
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

    args.do_train = True
    args.do_valid = True
    args.do_test = False
    data_parallel_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    train = PalmHDF5Dataset(args,
                        args.sub_train_lst[data_parallel_rank],
                        args.tokenizer,
                        args.vocab_words,
                        args.train_iters * args.gradient_accumulation_steps * args.batch_size // args.num_epochs,
                        is_training=True) 
    valid = Subset(train, list(range(args.eval_iters * eval_batch_size)))
    train = make_data_loader(train, batch_size, args) 
    valid = make_data_loader(valid, eval_batch_size, args)
    return (train, valid, None), tokenizer 

def PalmBatchify(batch):
    input_ids, input_mask, segment_ids, target_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = \
        list(), list(), list(), list(), list(), list(), list()
    for feature in batch:
        input_ids.append(torch.tensor(feature["input_ids"], dtype=torch.long))
        input_mask.append(torch.tensor(feature["input_mask"], dtype=torch.long))
        segment_ids.append(torch.tensor(feature["segment_ids"], dtype=torch.long))
        target_ids.append(torch.tensor(feature["target_ids"], dtype=torch.long))
        if "masked_lm_positions" in feature:
            masked_lm_positions.append(torch.tensor(feature["masked_lm_positions"], dtype=torch.long))
            masked_lm_ids.append(torch.tensor(feature["masked_lm_ids"], dtype=torch.long))
            masked_lm_weights.append(torch.tensor(feature["masked_lm_weights"], dtype=torch.float))


    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    segment_ids = torch.stack(segment_ids, 0)
    target_ids = torch.stack(target_ids, 0)
    if "masked_lm_positions" not in batch[0]:
        return {'text':input_ids, 'pad_mask':input_mask, 'types':segment_ids, 'target_ids':target_ids}
    else:
        masked_lm_positions = torch.stack(masked_lm_positions, 0)
        masked_lm_ids = torch.stack(masked_lm_ids, 0)
        masked_lm_weights = torch.stack(masked_lm_weights, 0)
    
        masked_lm_ids_ = torch.zeros(input_ids.size()).long()
        masked_lm_ids_.scatter_(1, masked_lm_positions, masked_lm_ids)
        masked_lm_ids_[:,0] = 0
        mask = torch.zeros(input_ids.size()).long().scatter(1, masked_lm_positions, 1)
        mask[:,0] = 0

        return {'text':input_ids, 'pad_mask':input_mask, 'types':segment_ids, 'target_ids':target_ids, 'masked_lm_positions':masked_lm_positions, 'mask_labels':masked_lm_ids_, 'mask':mask}

def batchify(batch):
    input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels = \
        list(), list(), list(), list(), list(), list(), list()
    for feature in batch:
        input_ids.append(torch.tensor(feature["input_ids"], dtype=torch.long))
        input_mask.append(torch.tensor(feature["input_mask"], dtype=torch.long))
        segment_ids.append(torch.tensor(feature["segment_ids"], dtype=torch.long))
        masked_lm_positions.append(torch.tensor(feature["masked_lm_positions"], dtype=torch.long))
        masked_lm_weights.append(torch.tensor(feature["masked_lm_weights"], dtype=torch.float))
        if "masked_lm_ids" in feature:
            masked_lm_ids.append(torch.tensor(feature["masked_lm_ids"], dtype=torch.long))
            next_sentence_labels.append(torch.tensor(feature["next_sentence_labels"], dtype=torch.long))
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    segment_ids = torch.stack(segment_ids, 0)
    masked_lm_positions = torch.stack(masked_lm_positions, 0)
    masked_lm_weights = torch.stack(masked_lm_weights, 0)
    masked_lm_ids = torch.stack(masked_lm_ids, 0)
    next_sentence_labels = torch.stack(next_sentence_labels, 0)
    masked_lm_ids_ = torch.zeros(input_ids.size()).long()
    masked_lm_ids_.scatter_(1, masked_lm_positions, masked_lm_ids)
    masked_lm_ids_[:,0] = 0
    mask = torch.zeros(input_ids.size()).long().scatter(1, masked_lm_positions, 1)
    mask[:,0] = 0
    return {'text':input_ids, 'pad_mask':input_mask, 'types':segment_ids, 'is_random':next_sentence_labels, 'mask_labels':masked_lm_ids_, 'mask':mask}

def create_instances_from_document(sent_to_doc, args, document_index, all_documents, vocab_words, rng, tokenizer):
    """Creates `TrainingInstance`s for a single document."""
    def split_doc(doc):
        #doc_piece = []
        #for index in range(doc.shape[0]):
        #    doc_piece.append(doc[index])
        #return doc_piece
        return doc
    document = all_documents[document_index][:-1]
    document_piece = split_doc(document)
    if args.dataset_has_lang_id:
        document_piece = document_piece[1:]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = args.seq_length - 3 if not args.one_sentence else args.seq_length - 2
    # We *usually* want to fill up the entire sequence since we are padding
    # to `seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < args.short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    current_chunk = []
    current_length = 0
    i = rng.randint(0, len(document_piece) - 1)
    while i < len(document_piece):
        segment = document_piece[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document_piece) - 1 or current_length >= target_seq_length:
            if current_chunk:
                ## @bao
                if args.one_sentence:
                    tokens_a = []
                    for chunk in current_chunk:
                        tokens_a.extend(chunk)
                    while len(tokens_a) > max_num_tokens:
                        if rng.random() < 0.5:
                            del tokens_a[0]
                        else:
                            tokens_a.pop()

                    tokens = []
                    segment_ids = []
                    tokens.append(tokenizer.vocab[args.cls_token])
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    tokens.append(tokenizer.vocab[args.sep_token])
                    segment_ids.append(0)
                    assert len(tokens) <= args.seq_length, "length of tokens is %d"%len(tokens)
                    
                    if rng.random() < 0.5:
                        if not args.no_nsp:
                            args.shuffle_order_prob = 1.0
                        else:
                            args.shuffle_order_prob = 0.0
                        next_sent_label = 1
                    else:
                        args.shuffle_order_prob = 0.0
                        next_sent_label = 0
                    (tokens, masked_lm_positions,
                    masked_lm_labels) = create_masked_lm_predictions(args, tokens, vocab_words, rng, tokenizer)
                    instance = TrainingInstance(
                        tokens=[x % len(tokenizer.vocab) for x in tokens],
                        segment_ids=segment_ids,
                        next_sent_label=next_sent_label,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=[x % len(tokenizer.vocab) for x in masked_lm_labels] if masked_lm_labels is not None else masked_lm_labels)
                    break

                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                if len(current_chunk) == 1 or rng.random() < 1.0/args.ns_type:
                    next_sent_label = 0
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = str(sent_to_doc[rng.randint(0, len(sent_to_doc) - 1)])
                        #random_document_index = str(rng.randint(0, len(all_documents) - 1))
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index][:-1]
                    random_document_piece = split_doc(random_document)
                    if args.dataset_has_lang_id:
                        random_document_piece = random_document_piece[1:]
                    random_start = rng.randint(0, len(random_document_piece) - 1)
                    for j in range(random_start, len(random_document_piece)):
                        tokens_b.extend(random_document_piece[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    if args.ns_type==2:
                        next_sent_label = 1
                    elif args.ns_type==3:
                        if rng.random() < 0.5:
                            next_sent_label = 1
                        else:
                            tokens_a, tokens_b = tokens_b, tokens_a
                            next_sent_label = 2
                    elif args.ns_type==5:
                        if len(current_chunk) >= 3 and rng.random() < 0.5:
                            p1 = rng.randint(1, len(current_chunk) - 1)
                            for _ in range(100):
                                p2 = rng.randint(1, len(current_chunk) - 1)
                                if p1 != p2:
                                    break
                            start = min(p1, p2)
                            end = max(p1, p2)
                            tokens_a = []
                            for item in current_chunk[:start]:
                                tokens_a.extend(item)
                            tokens_b = []
                            for item in current_chunk[end:]:
                                tokens_b.extend(item)
                            if rng.random() < 0.5:
                                next_sent_label = 3
                            else:
                                tokens_a, tokens_b = tokens_b, tokens_a
                                next_sent_label = 4
                        elif rng.random() < 0.5:
                            next_sent_label = 1
                        else:
                            tokens_a, tokens_b = tokens_b, tokens_a
                            next_sent_label = 2
                    else:
                        raise ValueError('Class type only support 2,3,5!')
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                assert len(tokens_a) >= 1, "length a: %d, length b: %d"%(len(tokens_a), len(tokens_b))
                assert len(tokens_b) >= 1, "length a: %d, length b: %d"%(len(tokens_a), len(tokens_b))

                tokens = []
                segment_ids = []
                tokens.append(tokenizer.vocab[args.cls_token])
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(tokenizer.vocab[args.sep_token])
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(tokenizer.vocab[args.sep_token])
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(args, tokens, vocab_words, rng, tokenizer)
                instance = TrainingInstance(
                    tokens=[x % len(tokenizer.vocab) for x in tokens],
                    segment_ids=segment_ids,
                    next_sent_label=next_sent_label,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=[x % len(tokenizer.vocab) for x in masked_lm_labels] if masked_lm_labels is not None else masked_lm_labels)
                break
        i += 1
    return instance

def create_masked_lm_predictions(args, tokens, vocab_words, rng, tokenizer):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    special_indexes = set()
    if not args.jieba:
        for (i, token) in enumerate(tokens):
            if token == tokenizer.vocab[args.cls_token] or token == tokenizer.vocab[args.sep_token]:
                special_indexes.add(i)
                continue
            if args.do_whole_word_mask and len(cand_indexes) >= 1 and not tokenizer.convert_ids_to_tokens(int(token)).startswith("▁") and not tokenizer.convert_ids_to_tokens(int(token)).startswith("<") and not tokenizer.convert_ids_to_tokens(int(token)).startswith("Ġ"): 
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
    
    else:
        raise NotImplementedError("Chinese wwm is not implemented.")        
 
    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # pylint: disable=invalid-name

    num_to_predict = min(args.max_preds_per_seq,
                       max(1, int(round(len(tokens) * args.masked_lm_prob))))
    if args.do_mask_rate_range:
        num_to_predict = rng.randint(1, num_to_predict)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            masked_lms = masked_lms[:num_to_predict]
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if args.all_token_mlm:
                masked_token = tokenizer.convert_tokens_to_ids([vocab_words[rng.randint(0, len(vocab_words) - 1)]])[0]
            else:
                if rng.random() < 0.8:
                    masked_token = tokenizer.vocab[args.mask_token]
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = tokenizer.convert_tokens_to_ids([vocab_words[rng.randint(0, len(vocab_words) - 1)]])[0]

            output_tokens[index] = masked_token

            token4predict = tokens[index]
            if args.predict_context_prob > 0.0:
                if rng.random() < args.predict_context_prob:
                    ## predict context word instead
                    if rng.random() < 0.5:
                        ## use left word
                        if index > 1:
                            token4predict = tokens[index - 1]
                    else:
                        ## use right word
                        if index < len(tokens) - 1:
                            token4predict = tokens[index + 1]
            masked_lms.append(masked_lm(index=index, label=token4predict))

            if args.continue_mask_prob > 0.0:
                for ii in [index - 1, index + 1]:
                    if (ii <= 0 or ii >= len(tokens) or ii in special_indexes 
                            or ii in covered_indexes or rng.random() > args.continue_mask_prob):
                        continue
                    covered_indexes.add(ii)
                    masked_token = None
                    # 80% of the time, replace with [MASK]
                    if args.all_token_mlm:
                        masked_token = tokenizer.convert_tokens_to_ids([vocab_words[rng.randint(0, len(vocab_words) - 1)]])[0]
                    else:
                        if rng.random() < 0.8:
                            masked_token = tokenizer.vocab[args.mask_token]
                        else:
                            # 10% of the time, keep original
                            if rng.random() < 0.5:
                                masked_token = tokens[ii]
                            # 10% of the time, replace with random word
                            else:
                                masked_token = tokenizer.convert_tokens_to_ids([vocab_words[rng.randint(0, len(vocab_words) - 1)]])[0]
                    output_tokens[ii] = masked_token
                    masked_lms.append(masked_lm(index=ii, label=tokens[ii]))

    if args.shuffle_order_prob > 0.0:
        cand_indexes = sum(cand_indexes, [])
        rng.shuffle(cand_indexes)
        if rng.random() < args.shuffle_order_prob:
            for idx in cand_indexes:
                if idx < len(output_tokens) - 2 and rng.random() < 0.1:
                    temp_tokens = output_tokens[idx:idx+3]
                    rng.shuffle(temp_tokens)
                    for i, token in enumerate(output_tokens[idx:idx+3]):
                        if temp_tokens[i] != token and idx+i not in covered_indexes:
                            masked_lms.append(masked_lm(index=idx+i, label=token))
                        covered_indexes.add(idx+i)
                    if len(masked_lms) >= args.max_preds_per_seq:
                        excess = len(masked_lms) - args.max_preds_per_seq
                        masked_lms = masked_lms[:args.max_preds_per_seq]
                        output_tokens[idx:idx+3] = temp_tokens[:3-excess] + output_tokens[idx:idx+excess]
                        break
                    output_tokens[idx:idx+3] = temp_tokens
    
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def convert_instance_to_feature(args, instance, tokenizer):
    input_ids = instance.tokens
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= args.seq_length

    while len(input_ids) < args.seq_length:
        if args.tokenizer_type.lower() in ['xlmr', 'roberta']:
            padding_id = 1
        elif args.tokenizer_type.lower() in ['bert']:
            padding_id = 0
        else:
            raise ValueError('{} tokenizer not supported'.format(args.tokenizer_type))
        input_ids.append(padding_id)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == args.seq_length
    assert len(input_mask) == args.seq_length
    assert len(segment_ids) == args.seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = instance.masked_lm_labels
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < args.max_preds_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    next_sentence_label = instance.next_sent_label
    
    feature = {}
    feature['input_ids'] = input_ids
    feature['input_mask'] = input_mask
    feature['segment_ids'] = segment_ids
    feature['masked_lm_positions'] = masked_lm_positions
    feature['masked_lm_ids'] = masked_lm_ids
    feature['masked_lm_weights'] = masked_lm_weights
    feature['next_sentence_labels'] = [next_sentence_label]

    return feature    

def get_few_shot_train_data(args, train_examples, labels):
    example_per_class = {}
    for label in labels:
        example_per_class[label] = []
    for example in train_examples:
        example_per_class[example.label].append(example)
    sampled_train_examples = []
    for label in labels:
        sampled_train_examples.extend(random.sample(example_per_class[label], args.few_shot_train_size_per_class))
    print_rank_0('len(sampled_train_examples): {}'.format(len(sampled_train_examples)))
    return sampled_train_examples


def data_preparation_nlu(tokenizer, processor, args):
    args.fast_train = False
    label_list = []
    if mpu.get_model_parallel_rank() == 0:
        dev_examples = processor.get_dev_examples(args.dev_data) if args.dev_data else []  # should be none []
        test_examples = processor.get_test_examples(args.test_data) if args.dev_data else [] # suppose to be []
        if len(dev_examples) == 0:
            train_examples, dev_examples, label_list = processor.train_dev_split(args.train_data)
        else:
            train_examples = processor.get_train_examples(args.train_data)
            label_list = processor.get_labels()
        if args.few_shot and  args.few_shot_train_size_per_class > 0:
            print_rank_0('Randomly sample a subset of train examples to train in few-shot learning setting')
            print_rank_0('few_shot_train_size: {}'.format(args.few_shot_train_size_per_class))
            train_examples = get_few_shot_train_data(args, train_examples, label_list)
        train_features = convert_examples_to_features(args, train_examples, [label_list], args.seq_length, tokenizer, 0, 1, True)
        dev_features = convert_examples_to_features(args, dev_examples, [label_list], args.seq_length, tokenizer, 0, 1, True)
        test_features = convert_examples_to_features(args, test_examples, [label_list], args.seq_length, tokenizer, 0, 1, False)
        train_dataset = FeatureDataset(train_features)
        dev_dataset = FeatureDataset(dev_features)
        test_dataset = FeatureDataset(test_features)
        data_size = torch.cuda.LongTensor([len(train_dataset), len(dev_dataset), len(test_dataset)])
        label_map = dict(enumerate(label_list))
        label_map_list = [label_map]
    else:
        train_dataset, dev_dataset, test_dataset = None, None, None
        data_size = torch.cuda.LongTensor([0, 0, 0])
        label_map_list = [None]

    torch.distributed.broadcast(data_size,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    torch.distributed.broadcast_object_list(label_map_list,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    
    args.num_of_classes = len(label_map_list[0].items())
    # rank 0 has had the label_list
    if len(label_list) == 0: 
        label_list = [None] * args.num_of_classes
        for key, value in label_map_list[0].items():
            label_list[key] = value
    args.data_size = data_size
    train_data, val_data, test_data = get_train_val_test_data_clean(args, train_dataset, dev_dataset, test_dataset)
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    args.train_iters = args.data_size[0].item() // (world_size * args.batch_size) * args.num_epochs
    return  train_data, val_data, test_data, label_list, args

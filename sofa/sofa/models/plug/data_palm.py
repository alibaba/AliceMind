import os
import json
import torch
import random
import logging
import h5py


import torch.distributed as dist
from torch.utils.data import Dataset, Subset
from multiprocessing import Pool

from .tokenization_plug import BertTokenizer
from .data_plug import get_train_val_test_data_clean
from sofa.utils import mpu


TRAIN_FILE_NAME = 'train.txt'
DEV_FILE_NAME = 'dev.txt'
TEST_FILE_NAME = 'test.txt'

# logger
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from .data_plug import InputExample, DataProcessor, \
                        create_instances_from_document, \
                        convert_instance_to_feature, \
                        make_data_loader, get_split\

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
                     'target_ids': feat.target_ids
                    }
        return feat_dict
        #return self.features[index]

    def lengths(self):
        return [len(feature.input_ids) for feature in self.features]

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, target_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_ids = target_ids

class NlpccKbqaProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'nlpcc2016-2018.kbqa.train')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'nlpcc2016.kbqa.test')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'nlpcc2016.kbqa.test')), "test")
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        target = None
        source = None
        guid = 0
        for line in lines:
            if 'triple' in line:
                target = line.strip().split()[-1]
            elif 'question' in line:
                source = line.strip().split('\t')[-1]
            else:
                target = None
                source = None
        
            if target and source:
                examples.append(
                    InputExample(guid=guid, text_a=source, text_b=target))
                guid += 1
        return examples 

class WeatherProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        print("data_dir", data_dir)
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "test")
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            source, target = line.strip().split("\t")
            examples.append(
                InputExample(guid=guid, text_a=source, text_b=target))
        return examples 

class YunProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "test")
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                source, target = line.strip().split("\t")
            except:
                print (line.strip())
                continue
            examples.append(
                InputExample(guid=guid, text_a=source, text_b=target))
        return examples 

class DianProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'train.txt')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, 'dev.txt')), "test")
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                source, target = line.strip().split("\t")
            except:
                print (line.strip())
                continue
            examples.append(
                InputExample(guid=guid, text_a=source, text_b=target))
        return examples 

class GenerationProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        if os.path.isfile(data_dir):
            file = data_dir
        else:
            file = os.path.join(data_dir, 'train.txt')
        return self._create_examples(
            self._read_txt(file), "train")

    def get_dev_examples(self, data_dir):
        if os.path.isfile(data_dir):
            file = data_dir
        else:
            file = os.path.join(data_dir, 'dev.txt')
        return self._create_examples(
            self._read_txt(file), "dev")
        
    def get_test_examples(self, data_dir):
        if os.path.isfile(data_dir):
            file = data_dir
        else:
            file = os.path.join(data_dir, 'test.txt')
        return self._create_examples(
            self._read_txt(file), "test")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                source, target = line.strip().split("\t")
            except:
                print (line.strip())
                continue
            examples.append(
                InputExample(guid=guid, text_a=source, text_b=target))
        return examples 

    def train_dev_split(self, train_data, ratio=0.8):
        lines = self._read_txt(train_data)
        random.shuffle(lines)
        train_example = self._create_examples(lines[:int((len(lines)+1)*ratio)], "train")
        dev_example = self._create_examples(lines[int((len(lines)+1)*ratio):], "dev")
        return train_example, dev_example

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
        seq_length = seq_length# * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length# * world_size
    split = get_split(args)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_model_type)
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

def convert_examples_to_features(args, examples, max_seq_length, tokenizer, is_training=False):

    features  = []
    for example in examples: 
        input_tokens = ["[CLS]"] + tokenizer.tokenize(example.text_a) + ["[SEP]"]
        if len(input_tokens) > max_seq_length:
            input_tokens = input_tokens[:max_seq_length-1] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        target_tokens = ["[CLS]"] + tokenizer.tokenize(example.text_b) + ["[SEP]"]
        if len(target_tokens) > args.tgt_length:
            target_tokens = target_tokens[:args.tgt_length-1] + ["[SEP]"]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        while len(target_ids) < args.tgt_length:
            target_ids.append(0)
            
        features.append(InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                target_ids=target_ids))
    return features

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids=False,
                               reset_attention_mask=False):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
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
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids

def data_preparation_nlg(tokenizer, processor, args):
    args.fast_train = False
    
    if mpu.get_model_parallel_rank() == 0:
        dev_examples = processor.get_dev_examples(args.dev_data) if args.dev_data else []  # should be none []
        test_examples = processor.get_test_examples(args.test_data) if args.test_data else [] # suppose to be []
        if len(dev_examples) == 0:
            train_examples, dev_examples = processor.train_dev_split(args.train_data)
        else:
            train_examples = processor.get_train_examples(args.train_data)
        train_features = convert_examples_to_features(args, train_examples, args.seq_length, tokenizer, True)
        dev_features = convert_examples_to_features(args, dev_examples, args.seq_length, tokenizer, True)
        test_features = convert_examples_to_features(args, test_examples, args.seq_length, tokenizer, False)
        train_dataset = FeatureDataset(train_features)
        dev_dataset = FeatureDataset(dev_features)
        test_dataset = FeatureDataset(test_features)
        data_size = torch.cuda.LongTensor([len(train_dataset), len(dev_dataset), len(test_dataset)])
    else:
        train_dataset, dev_dataset, test_dataset = None, None, None
        data_size = torch.cuda.LongTensor([0, 0, 0])
    train_data, val_data, test_data = get_train_val_test_data_clean(args, tokenizer, train_dataset, dev_dataset, test_dataset)
    torch.distributed.broadcast(data_size,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.data_size = data_size
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    args.train_iters = args.data_size[0].item() // (world_size * args.batch_size) * args.num_epochs
    return  train_data, val_data, test_data, [], args  


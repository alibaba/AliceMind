
"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""


from typing import Optional
from torch import Tensor

from PIL import Image
from dataset import vg_transforms as T


import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from models.tokenization_bert import BertTokenizer
from vgTools.utils.box_utils import xywh2xyxy


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

# Bert text encoding


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def collate_fn(raw_batch):

    raw_batch = list(zip(*raw_batch))

    img = torch.stack(raw_batch[0])
    img_mask = torch.tensor(np.array(raw_batch[1]))
    img_data = NestedTensor(img, img_mask)
    word_id = torch.tensor(np.array(raw_batch[2]))
    word_mask = torch.tensor(np.array(raw_batch[3]))
    text_data = NestedTensor(word_id, word_mask)
    bbox = torch.tensor(np.array(raw_batch[4]))

    batch = [img_data, text_data, bbox]
    return tuple(batch)


def collate_fn_val(raw_batch):

    raw_batch = list(zip(*raw_batch))

    img = torch.stack(raw_batch[0])
    img_mask = torch.tensor(np.array(raw_batch[1]))
    img_data = NestedTensor(img, img_mask)
    word_id = torch.tensor(np.array(raw_batch[2]))
    word_mask = torch.tensor(np.array(raw_batch[3]))
    text_data = NestedTensor(word_id, word_mask)
    bbox = torch.tensor(np.array(raw_batch[4]))
    raw_data = raw_batch[-1]
    batch = [img_data, text_data, bbox, raw_data]
    return tuple(batch)


class VisualGroundingDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
        'pailitao': {
            'splits': ('train', 'val', 'test')
        }
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128,
                 bert_model='bert-base-uncased', swin=False, odps_slice_tuple=None):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.return_idx = return_idx
        self.swin = swin
        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:  # refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]

        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]

        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        # assign item id as the last
        self.images = [item+(i,) for i, item in enumerate(self.images)]

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        # x1,y1,x2,y2
        if self.dataset == 'flickr':
            img_file, bbox, phrase, item_id = self.images[idx]
        elif self.dataset == 'pailitao':
            nid, title, crop_cord, image_url, item_id = self.images[idx]
            phrase = title
        else:
            img_file, _, bbox, phrase, attri, item_id = self.images[idx]
        # box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        image_url = 'none'

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox, item_id

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_data = {'idx': idx}
        from icecream import ic
        img, phrase, bbox, item_id = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        h, w = img.shape[-2:]
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']

        raw_data['phrase'] = phrase
        raw_data['gt_bbox'] = bbox
        raw_data['img'] = img
        raw_data['item_id'] = item_id

        # encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        if self.split == 'train':
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
        else:
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), raw_data


def make_transforms(args, image_set, is_onestage=False):
    imsize = args['image_res']
    if image_set == 'train':
        scales = []
        if args['aug_scale']:
            rate = imsize//20  
            for i in range(7):
                scales.append(imsize - rate * i)
        else:
            scales = [imsize]

        if args['aug_crop']:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args['aug_blur']),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, mean=(0.48145466, 0.4578275, 0.40821073), std=(
                0.26862954, 0.26130258, 0.27577711), aug_translate=args['aug_translate'])
        ])

    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')



def build_vg_dataset(split, args, dataset_name=None):
    return VisualGroundingDataset(data_root=args['data_root'],
                                  split_root=args['split_root'],
                                  dataset=dataset_name,
                                  split=split,
                                  bert_model=args['text_encoder'],
                                  transform=make_transforms(args, split),
                                  max_query_len=args['max_query_len'],
                                  )


def build_uni_training_dataset(args):
    from torch.utils import data
    datasets = []

    for dataset_name in ['referit', 'unc', 'unc+', 'gref_umd']:
        max_query_len = 20 if 'gref' not in dataset_name else 20 
        datasets.append(VisualGroundingDataset(data_root=args['data_root'],
                        split_root=args['split_root'],
                        dataset=dataset_name,
                        split='train',
                        bert_model=args['text_encoder'],
                        transform=make_transforms(args, 'train'),
                        max_query_len=max_query_len))
    uni_dataset = data.ConcatDataset(datasets)
    return uni_dataset

# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
import logging
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
import h5py
import base64
import glob

from PIL import Image
import torchvision.transforms as transforms

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014.tsv',
    'valid': 'val2014.tsv',
    'minival': 'val2014.tsv',
    'nominival': 'val2014.tsv',
    'test': 'test2015.tsv',
}

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        logger.info("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset
        self.add_box_size = args.add_box_size

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        if args.use_vit:
            self.imgid2img = {}
            self.image_dir_list = args.image_hdf5_file.split(',')
            for image_dir in self.image_dir_list:
                image_npz_files = os.path.join(image_dir, '*.jpg')
                for image_npz_file in glob.glob(image_npz_files):
                    image_id = image_npz_file.split('/')[-1].split('.')[0]
                    self.imgid2img[image_id] = image_npz_file
            logger.info('total image number is: {}'.format(len(self.imgid2img)))
        elif args.use_hdf5:
            self.imgid2img = {}
            self.image_file_list = args.image_hdf5_file.split(',')
            for i, image_file in enumerate(self.image_file_list):
                with h5py.File(image_file, 'r') as all_images:
                    for image_id in all_images.keys():
                        self.imgid2img[image_id] = i
            logger.info('total image number is: {}'.format(len(self.imgid2img)))
        elif args.use_npz:
            self.imgid2img = {}
            self.image_dir_list = args.image_hdf5_file.split(',')
            for image_dir in self.image_dir_list:
                image_npz_files = os.path.join(image_dir, '*.npz')
                for image_npz_file in glob.glob(image_npz_files):
                    image_id = image_npz_file.split('/')[-1].split('.')[0]
                    self.imgid2img[image_id] = image_npz_file
            logger.info('total image number is: {}'.format(len(self.imgid2img)))
        else:
            img_data = []
            for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/model-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(
                    load_obj_tsv(os.path.join(MSCOCO_IMGFEAT_ROOT, SPLIT2NAME[split]), topk=load_topk))

            # Convert img list to dict
            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        logger.info("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        if args.use_hdf5:
            file_idx = self.imgid2img[img_id]
            with h5py.File(self.image_file_list[file_idx], 'r') as all_images:
                img_info = all_images[img_id]
                img_h, img_w, obj_num = np.frombuffer(base64.b64decode(img_info[0]), dtype=np.int64).tolist()
                feats = np.frombuffer(base64.b64decode(img_info[6]), dtype=np.float32).reshape((obj_num, -1)).copy()
                boxes =  np.frombuffer(base64.b64decode(img_info[5]), dtype=np.float32).reshape((obj_num, 4)).copy()
        elif args.use_npz:
            file_path = self.imgid2img[img_id]
            img_info = np.load(file_path)
            img_h = img_info['img_h']
            img_w = img_info['img_w']
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
        else:
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']


        if args.padding:
            if obj_num != args.max_objects:
                align_obj_num = args.max_objects
                feat_len = np.size(feats, 1)
                box_len = np.size(boxes, 1)
                feats = np.row_stack((feats, np.zeros((align_obj_num - obj_num, feat_len), dtype=np.float32)))
                boxes = np.row_stack((boxes, np.zeros((align_obj_num - obj_num, box_len), dtype=np.float32)))
                obj_num = align_obj_num

        assert obj_num == len(boxes) == len(feats)

        boxes = boxes.copy()
        if args.use_struct_emb:

            resize_ratio = args.max_2d_position_embeddings * 1.0 / max(img_w, img_h)
            boxes *= resize_ratio
            boxes = boxes.astype(int)

        else:
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            if self.add_box_size:
                box_size = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = np.column_stack((boxes, box_size.reshape((-1, 1))))
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)



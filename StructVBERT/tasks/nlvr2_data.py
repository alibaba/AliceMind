# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
from torch.utils.data import Dataset
import logging

from param import args
from utils import load_obj_tsv
import h5py
import base64
import glob
import os

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NLVR2Dataset:
    """
    An NLVR2 data example in json file:
    {
        "identifier": "train-10171-0-0",
        "img0": "train-10171-0-img0",
        "img1": "train-10171-0-img1",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.
        ",
        "uid": "nlvr2_train_0"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/nlvr2/%s.json" % split)))
        logger.info("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['uid']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class NLVR2TorchDataset(Dataset):
    def __init__(self, dataset: NLVR2Dataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        if args.use_hdf5:
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
            if 'train' in dataset.splits:
                img_data.extend(load_obj_tsv('data/nlvr2_imgfeat/nlvr2_train.tsv', topk=topk))
            if 'valid' in dataset.splits:
                img_data.extend(load_obj_tsv('data/nlvr2_imgfeat/nlvr2_valid.tsv', topk=topk))
            if 'test' in dataset.name:
                img_data.extend(load_obj_tsv('data/nlvr2_imgfeat/nlvr2_test.tsv', topk=topk))
            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                self.data.append(datum)
        logger.info("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        ques_id = datum['uid']
        ques = datum['sent']

        # Get image info
        boxes2 = []
        feats2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            if args.use_hdf5:
                file_idx = self.imgid2img[img_id]
                with h5py.File(self.image_file_list[file_idx], 'r') as all_images:
                    img_info = all_images[img_id]
                    img_h, img_w, obj_num = np.frombuffer(base64.b64decode(img_info[0]), dtype=np.int64).tolist()
                    feats = np.frombuffer(base64.b64decode(img_info[6]), dtype=np.float32).reshape((obj_num, -1)).copy()
                    boxes = np.frombuffer(base64.b64decode(img_info[5]), dtype=np.float32).reshape((obj_num, 4)).copy()
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
                boxes = img_info['boxes'].copy()
                feats = img_info['features'].copy()
                obj_num = img_info['num_boxes']
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

            # Normalize the boxes (to 0 ~ 1)
            boxes[..., (0, 2)] /= img_w
            boxes[..., (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

            boxes2.append(boxes)
            feats2.append(feats)
        feats = np.stack(feats2)
        boxes = np.stack(boxes2)

        # Create target
        if 'label' in datum:
            label = datum['label']
            return ques_id, feats, boxes, ques, label
        else:
            return ques_id, feats, boxes, ques


class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer

        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))


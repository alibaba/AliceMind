import os
import json
import random
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from dataset.utils import pre_question

import decord
from decord import VideoReader

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

decord.bridge.set_bridge("torch")


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class videoqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', max_img_size=384, read_local_data=True, num_frm=16, frm_sampling_strategy='uniform'):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += load_jsonl(f)

        self.transform = transform
        self.video_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.read_local_data = read_local_data

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_fmt = '.mp4'

        if split=='test':
        # if True:
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = list(json.load(open(answer_list,'r')).keys())


        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        for idx, ann in enumerate(self.ann):
            ann['question_id'] = idx
        self.gts = {x['question_id']: x for x in self.ann}
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]

        video_path = os.path.join(self.video_root, ann['video_id'] + self.video_fmt)
        # print(video_path)
        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        
        return video, ann['question'], ann['question_id']

    
    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            print(e)
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms

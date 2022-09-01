import os
import json
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from dataset.utils import pre_question

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, gqa_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', read_local_data=True, add_ocr=False, add_object=False):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.gqa_root = gqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.read_local_data = read_local_data
        self.add_ocr = add_ocr
        self.add_object = add_object
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    
        if self.add_ocr:
            self.max_ques_words = 30
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
        elif ann['dataset']=='gqa':
            image_path = os.path.join(self.gqa_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        question = ann['question']
        if self.add_ocr and "ocr" in ann:
            ocrs = ann['ocr']
            ocr_tokens = []
            poses = []
            for ocr in ocrs:
                pos, token = ocr
                ocr_tokens.append(token)
                poses.append(pos)
            if len(ocr_tokens) > 0:
                ocr_string = pre_question(" ".join(ocr_tokens), self.max_ques_words)
                question = question + " [SEP] " + ocr_string
        if self.add_object and "object_label" in ann:
            objects = ann["object_label"]
            question = question + " [SEP] " + " ".join(objects.split("&&"))
        # question = pre_question(question,self.max_ques_words)   
        if self.split == 'test':
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            
            if ann['dataset']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.5]  
            elif ann['dataset']=='gqa':
                answers = [ann['answer']]
                weights = [0.5]  

            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights

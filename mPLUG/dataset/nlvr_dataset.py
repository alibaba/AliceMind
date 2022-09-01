import json
import random
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        if "train" in ann['images'][0]:
            ann['images'][0] = 'train/'+ann['images'][0].split("/")[-1]
            ann['images'][1] = 'train/'+ann['images'][1].split("/")[-1]
        image0_path = os.path.join(self.image_root,ann['images'][0])        
        image0 = Image.open(image0_path).convert('RGB')   
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['images'][1])              
        image1 = Image.open(image1_path).convert('RGB')     
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['sentence'], self.max_words)
        
        if ann['label']=='True':
            label = 1
        else:
            label = 0

        words = sentence.split(' ')
        
        if 'left' not in words and 'right' not in words:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                return image1, image0, sentence, label
        else:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                new_words = []
                for word in words:
                    if word=='left':
                        new_words.append('right')
                    elif word=='right':
                        new_words.append('left')        
                    else:
                        new_words.append(word)                    
                        
                sentence = ' '.join(new_words)
                return image1, image0, sentence, label


'''
 * Copyright (c) 2021, salesforce.com, inc.
 * Copyright (c) 2010-2015 Alibaba Group Holding Limited.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from pickle import NONE, TRUE
from turtle import forward

from matplotlib.transforms import Transform
from models.vit import VisionTransformer, interpolate_pos_embed
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel,BertEncoder, BertPrefixModelForGrounding, FusionModel
from models.visual_transformers import initialize_clip

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from einops import repeat
from vgTools.utils.box_utils import xywh2xyxy,generalized_box_iou
from icecream import ic
from mmdet.models.losses import smooth_l1_loss
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MPLUG(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True,
                 ):
        super().__init__()
        import copy
        config = copy.deepcopy(config)
        self.config = config
        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']

        self.visual_encoder, _ = initialize_clip(config)
        for module in self.visual_encoder.modules():
            from models.clip.model import Transformer
            if isinstance(module,Transformer):
                module.use_checkpoint = (config['vision_width'] == 1024)

        vision_width = config['vision_width']
        
        self.module_setting(config)
        self.config_encoder.gradient_checkpointing = False
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=self.config_encoder, add_pooling_layer=False)  
        self.fusion_encoder = FusionModel.from_pretrained(text_encoder, config=self.config_fusion, add_pooling_layer=False)  
        
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(text_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.large = False
        if self.config_encoder.hidden_size != vision_width:
            self.visn_fc = nn.Linear(vision_width, self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.text_decoder = BertPrefixModelForGrounding(config=self.config_decoder)
     
        config['image_res']=336
        prefix_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[unused339]']))
        target_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids([f'[unused{340+_}]' for _ in range(config['image_res']+1)]))
        self.register_buffer('prefix_ids',prefix_ids)
        self.register_buffer('target_ids',target_ids)
    
        bbox_prob_mask = torch.zeros(len(self.tokenizer))
        bbox_prob_mask[self.target_ids[0]:self.target_ids[-1]+1]=1
        bbox_prob_mask = (1.0 - bbox_prob_mask) * -10000.0
        self.register_buffer('bbox_prob_mask',bbox_prob_mask)
        self.weight_decoder = True

        self.del_unused_modules()
    
    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])   
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
    
    def del_unused_modules(self,):
        self.text_encoder.encoder.layer = self.text_encoder.encoder.layer[:self.config_encoder.stride_layer]
        pass


    def forward(self, image_data, text_data, extra:dict):
        bs = image_data.tensors.shape[0]

        image = image_data.tensors
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
  
        text = text_data.tensors
        text_mask = text_data.mask
        text_output = self.text_encoder(text, attention_mask=text_mask,
                                             return_dict=True)
        text_embeds = text_output.last_hidden_state
   
        image_output, text_output, = self.fusion_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                    )
        text_output = torch.cat([image_output,text_output],1)
        merge_text_attention = torch.cat([image_atts, text_mask], 1)
        prefix_ids = repeat(self.prefix_ids,'D -> B D',B=bs)
        
        if 'targets' in extra:
            answer_targets = self.process_target(extra['targets'])
            prefix_ids = torch.cat([prefix_ids,answer_targets],dim=1)
            answer_output = self.text_decoder(prefix_ids,  
                                        encoder_hidden_states = text_output,
                                        encoder_attention_mask = merge_text_attention,                  
                                        labels = prefix_ids,
                                        return_dict = True,   
                                        soft_labels = None,
                                        reduction = 'none',
                                        prob_mask=self.bbox_prob_mask
                                        )   
            
            loss_seq = answer_output.loss
            if self.weight_decoder:
                with torch.no_grad():
                    pred_box = self.process_bbox(torch.argmax(answer_output.logits[:, :-1, :].contiguous(),dim=-1))
                    weight_bbox = F.l1_loss(pred_box, extra['targets'], reduction='none').clamp(0,5)
                    weight_giou = 1 - torch.diag(generalized_box_iou(
                        xywh2xyxy(pred_box),
                        xywh2xyxy(extra['targets'])
                    ))
                loss_seq = loss_seq.view(bs,-1,4)
                loss_seq = loss_seq * weight_bbox
                loss_seq = loss_seq * weight_giou.unsqueeze(1)
            
            loss_seq = loss_seq.mean()
            return {'loss_seq':loss_seq}
        else:
            # Eval model
            answer_targets = None
            past_key_values = None
            for _ in range(4):
                answer_output = self.text_decoder(prefix_ids,  
                                        encoder_hidden_states = text_output,
                                        encoder_attention_mask = merge_text_attention,                  
                                        labels = None,
                                        past_key_values = None,
                                        return_dict = True,   
                                        soft_labels = None,
                                        reduction = 'none',
                                        prob_mask=self.bbox_prob_mask
                                        )
                prefix_ids = torch.cat([prefix_ids,torch.argmax(answer_output.logits[:,-1,:],dim=-1).unsqueeze(1)],dim=1)

            pred_box = self.process_bbox(prefix_ids[:,1:])
            
            return pred_box

    @torch.no_grad()
    def process_bbox(self,bbox):
        bbox = bbox - self.target_ids[0]
        bbox = torch.true_divide(bbox, self.config['image_res'])
        assert torch.all(bbox<=1)
        return bbox

    @torch.no_grad()
    def process_target(self,targets):

        targets = targets.clamp(0,1)
        targets = (targets * self.config['image_res']).floor().long()

        assert torch.all(targets<=self.config['image_res'])
        targets = targets + self.target_ids[0]
        return targets


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


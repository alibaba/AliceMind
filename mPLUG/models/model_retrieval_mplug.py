from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertLMHeadModel, FusionModel
from models.visual_transformers import initialize_clip

import torch
from torch import nn
import torch.nn.functional as F


class MPLUG(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.module_setting(config)
        self.visual_encoder, _ = initialize_clip(config)
        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
        self.init_distill(config)

    def forward(self, image, text, alpha, idx):

        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
            if self.large:
                image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            if self.distill:
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        if self.distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        ###=================================###
        # forward the positve image-text pair
        _, output_pos = self.fusion_encoder(encoder_embeds=text_embeds,
                                            attention_mask=text.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=False,
                                            )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

            # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        _, output_neg = self.fusion_encoder(encoder_embeds=text_embeds_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=False,
                                            )

        vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm

    def module_setting(self, config):
        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.text_width = self.config_encoder.hidden_size
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True

        self.vision_proj = nn.Linear(self.text_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.itm_head = nn.Linear(self.text_width, 2)

        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def init_distill(self, config):
        self.distill = config['distill']
        if self.distill:
            self.visual_encoder_m, _ = initialize_clip(config)
            self.text_encoder_m = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
            self.fusion_encoder_m = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
            self.vision_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                [self.vision_proj, self.vision_proj_m],
                               ]
            if self.config_encoder.hidden_size != config['vision_width']:
                self.visn_fc_m = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend([[self.visn_fc, self.visn_fc_m], [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params()
            self.momentum = 0.995

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


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
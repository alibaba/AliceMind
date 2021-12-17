import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Callable
from model.Bert import BertModel

class PruneBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, 
        config, 
        contrastive_temperature, 
        ce_loss_weight, 
        cl_unsupervised_loss_weight, 
        cl_supervised_loss_weight, 
        distill_loss_weight, 
        extra_examples, 
        alignrep,
        get_teacher_logits,
        distill_temperature,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        assert alignrep in ['mean-pooling', 'cls']
        self.alignrep = alignrep

        self.contrastive_temperature = contrastive_temperature
        self.ce_loss_weight = ce_loss_weight
        self.cl_unsupervised_loss_weight = cl_unsupervised_loss_weight
        self.cl_supervised_loss_weight = cl_supervised_loss_weight
        self.extra_examples = extra_examples

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # mask: 1 -> keep 0 -> discard
        self.register_buffer('head_mask', torch.ones(config.num_hidden_layers, config.num_attention_heads))
        self.register_buffer('intermediate_mask', torch.ones(config.num_hidden_layers, config.intermediate_size))

        self.init_weights()

        # contrastive 
        self.global_representations_bank_finetuned = None
        self.global_representations_bank_pretrained = None
        self.global_representations_bank_snaps = None
        self.global_labels_bank = None

        # distillation
        self.distill_loss_weight = distill_loss_weight
        self.get_teacher_logits = get_teacher_logits # a func
        self.distill_temperature = distill_temperature

    def forward(
        self,
        idx=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        intermediate_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encode_example=False,
    ):

        # In most cases the head_mask and intermediate mask would be None
        # so that the model will use the masks on its own
        # One exception is when we calcuate importance score

        if head_mask is None:
            head_mask = self.head_mask
        if intermediate_mask is None:
            intermediate_mask = self.intermediate_mask

        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            intermediate_mask=intermediate_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_output, pooled_output = outputs[0], outputs[1]

        # just for encode example
        if encode_example:
            if self.alignrep == 'mean-pooling':
                result = torch.sum(all_output * (attention_mask.unsqueeze(-1) == 1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            else:
                result = pooled_output
            result = torch.nn.functional.normalize(result, p=2, dim=-1)
            return result # bsz * hidden_state

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # cross entropy loss
        loss *= self.ce_loss_weight

        # Distillation loss
        if self.get_teacher_logits is not None:
            teacher_logits = self.get_teacher_logits(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).logits

            loss_logits = (
                F.kl_div(
                    input=F.log_softmax(logits / self.distill_temperature, dim=-1),
                    target=F.softmax(teacher_logits / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )

            loss += self.distill_loss_weight * loss_logits

        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_pretrained, self.global_labels_bank, all_output, pooled_output, attention_mask, labels)
        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_finetuned, self.global_labels_bank, all_output, pooled_output, attention_mask, labels)
        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_snaps, self.global_labels_bank, all_output, pooled_output, attention_mask, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calculate_contrastive_loss(
        self, 
        idx, 
        global_representations_bank, 
        global_labels_bank,
        all_output, 
        pooled_output, 
        attention_mask,
        labels
    ):
        loss = 0
        if idx is not None and global_representations_bank is not None:
            # representations: bsz * rep_num * hidden_state
            # labels: bsz
            representations = torch.index_select(global_representations_bank, dim=0, index=idx.cpu()).to(pooled_output)
            bsz, rep_num, hid_size = representations.size()

            if self.alignrep == 'mean-pooling':
                pooled_output = torch.sum(all_output * (attention_mask.unsqueeze(-1) == 1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            else:
                pooled_output = pooled_output
            pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1) # bsz * hidden_state
            
            # also add current data
            representations = torch.cat((pooled_output.unsqueeze(1).detach(), representations), dim=1) # bsz * (rep_num+1) * hidden_state
            representations = representations.reshape(bsz*(rep_num+1), hid_size)

            # sample more examples
            extra = self.extra_examples // global_representations_bank.size(1) 
            extra_idx = torch.LongTensor(random.sample(range(global_representations_bank.size(0)), k=extra))
            extra_labels = torch.index_select(global_labels_bank, dim=0, index=extra_idx).to(pooled_output) # extra
            extra_representations = torch.index_select(global_representations_bank, dim=0, index=extra_idx).view(-1, hid_size).to(pooled_output) # (extra * rep_num) * hidden_state
            extra_idx = extra_idx.to(pooled_output)

            representations = torch.cat((representations, extra_representations), dim=0) 
            contrastive_score = torch.mm(pooled_output, representations.t()) # bsz * (bsz*(rep_num+1)+(extra * rep_num))

            # exclude choosing myself
            # contrastive_mask: choosing myself -> 1，e.g., contrastive_mask[0,0] = contrastive_mask[1, rep_num] = contrastive_mask[2, 2*rep_num] = 1
            contrastive_mask = torch.unbind(torch.eye(bsz).to(contrastive_score), dim=1)
            contrastive_mask = [torch.cat((m.unsqueeze(1), torch.zeros(bsz, rep_num).to(contrastive_score)), dim=1) for m in contrastive_mask]
            contrastive_mask = torch.cat(contrastive_mask, dim=1) # bsz * (bsz*(rep_num+1))
            contrastive_mask = torch.cat((contrastive_mask, torch.zeros(bsz, extra * rep_num).to(contrastive_mask)), dim=1) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            contrastive_score /= self.contrastive_temperature
            contrastive_score = contrastive_score.masked_fill(contrastive_mask==1, -1e6)
            contrastive_score = torch.nn.functional.log_softmax(contrastive_score, dim=-1)

            # calculate unsupervised_mask, only maintain the positive positives (belonging to the same instance) log_softmax
            all_idx = torch.cat((idx.unsqueeze(1).repeat(1, rep_num+1).view(-1), extra_idx.unsqueeze(1).repeat(1, rep_num).view(-1)), dim=0) # (bsz*(rep_num+1)+(extra * rep_num))
            all_idx = all_idx.unsqueeze(0).expand(bsz, -1) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            positive_mask = (idx.unsqueeze(1) == all_idx) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            mask_contrastive_score = contrastive_score.masked_fill( (positive_mask==0) | (contrastive_mask==1), 0)
            positive_num = torch.sum(positive_mask, dim=1, keepdim=True) - 1
            loss += - self.cl_unsupervised_loss_weight * torch.sum(mask_contrastive_score / positive_num) / torch.sum(mask_contrastive_score!=0)
            
            # supervised_mask
            all_labels = torch.cat((labels.unsqueeze(1).repeat(1, rep_num+1).view(-1), extra_labels.unsqueeze(1).repeat(1, rep_num).view(-1)), dim=0) # (bsz*(rep_num+1)+(extra * rep_num))
            all_labels = all_labels.unsqueeze(0).expand(bsz, -1) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            positive_mask = (labels.unsqueeze(1) == all_labels) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            mask_contrastive_score = contrastive_score.masked_fill( (positive_mask==0) | (contrastive_mask==1), 0)
            positive_num = torch.sum(positive_mask, dim=1, keepdim=True) - 1
            loss += - self.cl_supervised_loss_weight * torch.sum(mask_contrastive_score / positive_num) / torch.sum(mask_contrastive_score!=0)
        
        return loss
        
class PruneBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, 
        config, 
        contrastive_temperature, 
        ce_loss_weight, 
        cl_unsupervised_loss_weight, 
        distill_loss_weight, 
        extra_examples, 
        alignrep,
        get_teacher_logits,
        distill_temperature
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        assert alignrep in ['mean-pooling', 'cls']
        self.alignrep = alignrep

        self.contrastive_temperature = contrastive_temperature
        self.ce_loss_weight = ce_loss_weight
        self.cl_unsupervised_loss_weight = cl_unsupervised_loss_weight
        self.extra_examples = extra_examples

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # mask: 1 -> keep 0 -> discard
        self.register_buffer('head_mask', torch.ones(config.num_hidden_layers, config.num_attention_heads))
        self.register_buffer('intermediate_mask', torch.ones(config.num_hidden_layers, config.intermediate_size))

        self.init_weights()

        # contrastive 
        self.global_representations_bank_finetuned = None
        self.global_representations_bank_pretrained = None
        self.global_representations_bank_snaps = None

        # distillation
        self.distill_loss_weight = distill_loss_weight
        self.get_teacher_logits = get_teacher_logits
        self.distill_temperature = distill_temperature

    def forward(
        self,
        idx=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        intermediate_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encode_example=False,
    ):

        # In most cases the head_mask and intermediate mask would be None
        # so that the model will use the masks on its own
        # One exception is when we calcuate importance score
        
        if head_mask is None:
            head_mask = self.head_mask
        if intermediate_mask is None:
            intermediate_mask = self.intermediate_mask

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            intermediate_mask=intermediate_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_output = outputs[0]
        pooled_output = all_output[:, 0, :]

        # just for encode example
        if encode_example:
            if self.alignrep == 'mean-pooling':
                result = torch.sum(all_output * (attention_mask.unsqueeze(-1) == 1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            else:
                result = pooled_output
            result = torch.nn.functional.normalize(result, p=2, dim=-1)
            return result # bsz * hidden_state

        logits = self.qa_outputs(all_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = 0
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        # cross entropy loss
        loss *= self.ce_loss_weight

        # Distillation loss
        if self.get_teacher_logits is not None:
            teacher_logits = self.get_teacher_logits(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            teacher_start_logits, teacher_end_logits = teacher_logits.start_logits, teacher_logits.end_logits

            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits / self.distill_temperature, dim=-1),
                    target=F.softmax(teacher_start_logits / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits / self.distill_temperature, dim=-1),
                    target=F.softmax(teacher_end_logits / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            loss_logits = (loss_start + loss_end) / 2.0

            loss += self.distill_loss_weight * loss_logits

        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_pretrained, all_output, pooled_output, attention_mask)
        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_finetuned, all_output, pooled_output, attention_mask)
        loss += self.calculate_contrastive_loss(idx, self.global_representations_bank_snaps, all_output, pooled_output, attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        loss = loss if start_positions is not None and end_positions is not None else None
        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def calculate_contrastive_loss(
        self,
        idx, 
        global_representations_bank, 
        all_output, 
        pooled_output, 
        attention_mask,
    ):
        loss = 0
        if idx is not None and global_representations_bank is not None:
            # representations: bsz * rep_num * hidden_state
            representations = torch.index_select(global_representations_bank, dim=0, index=idx.cpu()).to(pooled_output)
            bsz, rep_num, hid_size = representations.size()

            if self.alignrep == 'mean-pooling':
                pooled_output = torch.sum(all_output * (attention_mask.unsqueeze(-1) == 1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            else:
                pooled_output = pooled_output
            pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1) # bsz * hidden_state
            
            # also add current data
            representations = torch.cat((pooled_output.unsqueeze(1).detach(), representations), dim=1) # bsz * (rep_num+1) * hidden_state
            representations = representations.reshape(bsz*(rep_num+1), hid_size)

            # sample more examples
            extra = self.extra_examples // global_representations_bank.size(1) 
            extra_idx = torch.LongTensor(random.sample(range(global_representations_bank.size(0)), k=extra))
            extra_representations = torch.index_select(global_representations_bank, dim=0, index=extra_idx).view(-1, hid_size).to(pooled_output) # (extra * rep_num) * hidden_state
            extra_idx = extra_idx.to(pooled_output)

            representations = torch.cat((representations, extra_representations), dim=0) 
            contrastive_score = torch.mm(pooled_output, representations.t()) # bsz * (bsz*(rep_num+1)+(extra * rep_num))

            # exclude choosing myself
            # contrastive_mask: choosing myself -> 1，e.g., contrastive_mask[0,0] = contrastive_mask[1, rep_num] = contrastive_mask[2, 2*rep_num] = 1
            contrastive_mask = torch.unbind(torch.eye(bsz).to(contrastive_score), dim=1)
            contrastive_mask = [torch.cat((m.unsqueeze(1), torch.zeros(bsz, rep_num).to(contrastive_score)), dim=1) for m in contrastive_mask]
            contrastive_mask = torch.cat(contrastive_mask, dim=1) # bsz * (bsz*(rep_num+1))
            contrastive_mask = torch.cat((contrastive_mask, torch.zeros(bsz, extra * rep_num).to(contrastive_mask)), dim=1) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            contrastive_score /= self.contrastive_temperature
            contrastive_score = contrastive_score.masked_fill(contrastive_mask==1, -1e6)
            contrastive_score = torch.nn.functional.log_softmax(contrastive_score, dim=-1)

            # calculate unsupervised_mask, only maintain the positive positives (belonging to the same instance) log_softmax
            all_idx = torch.cat((idx.unsqueeze(1).repeat(1, rep_num+1).view(-1), extra_idx.unsqueeze(1).repeat(1, rep_num).view(-1)), dim=0) # (bsz*(rep_num+1)+(extra * rep_num))
            all_idx = all_idx.unsqueeze(0).expand(bsz, -1) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            positive_mask = (idx.unsqueeze(1) == all_idx) # bsz * (bsz*(rep_num+1)+(extra * rep_num))
            mask_contrastive_score = contrastive_score.masked_fill( (positive_mask==0) | (contrastive_mask==1), 0)
            positive_num = torch.sum(positive_mask, dim=1, keepdim=True) - 1
            loss += - self.cl_unsupervised_loss_weight * torch.sum(mask_contrastive_score / positive_num) / torch.sum(mask_contrastive_score!=0)
        
        return loss

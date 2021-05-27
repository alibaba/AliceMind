import copy

import torch
import torch.nn as nn
from others.transformers import BertModel, BertConfig
from others.transformers import RobertaModel, RobertaConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    #if checkpoint is not None:
    if False:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    #if checkpoint is not None:
    if False:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    #if checkpoint is not None:
    if False:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim



class SpanPositionalEncoding():

    def __init__(self, itos, padding_idx):
        self.itos = itos
        self.padding_idx = padding_idx

    def __call__(self, x):
        # print(x.shape)
        seq_len, batch_size = x.shape
        pos_enc = torch.zeros_like(x)

        for b in range(batch_size):
            for i in range(seq_len):
                if '##' not in self.itos[x[i, b].item()] or i == 0:
                    pos_enc[i, b] = 0
                else:
                    if i > 0 and x[i, b].item() != self.padding_idx and '##' in self.itos[x[i, b].item()]:
                        pos_enc[i, b] = 1 + pos_enc[i-1, b]
                    else:
                        pos_enc[i, b] = 0
        return pos_enc

    @staticmethod
    def decay(pos, prev, curr, decay_rate=0.1):
        """Copy rate decaying for current step.

        Arguments:
            pos {[type]} -- [S, B]
            prev {[type]} -- copy rate for last step, [T-1, B, S]
            curr {[type]} -- copy rate for current step,  [T, B, S]

        Keyword Arguments:
            decay_rate {float} -- [description] (default: {0.1})

        Returns:
            [type] -- new copy rate for current step, [T, B, S]
        """
        steps = curr.size(0)
        print ('curr_shape: ', curr.shape)
        print ('prev_shape: ', prev.shape)
        print ('pos_shape: ', pos.shape)
        residual = torch.zeros_like(curr)  # [T, B, S]
        mask = torch.zeros_like(curr)  # [T, B, S]

        residual[1:, ..., 1:] += prev[..., :-1] * decay_rate  # [T, B, S]
        # Only if the current step is within the same span of the last step.
        flag = (pos[1:] > pos[:-1]).float()  # [S-1, B]
        mask[-1:, ..., 1:] += flag.transpose(0, 1).unsqueeze(0).repeat([1, 1, 1])
        mask = (mask == 1.0)
        new = residual + (1 - decay_rate) * curr
        ans = torch.where(mask, new, curr)
        return torch.softmax(ans, dim=-1)

def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class CopyGenerator(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(CopyGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.gen_proj = nn.Linear(d_model, vocab_size)
        self.prob_proj = nn.Linear(d_model*2, 1)
        self.sig_proj = nn.Sigmoid()
        self.gen_softmax = nn.Softmax(dim=-1)
        self.copy_softmax = nn.Softmax(dim=-1)
    def forward(self, src, decode_output, decode_attn, memory):
        decode_attn = torch.mean(decode_attn, dim=1)
        batch_size, steps, seq = decode_attn.size()
        src = src.unsqueeze(1).repeat([1, steps, 1])
        # vocab
        gen_logits = self.gen_proj(decode_output)
        copy_logits = torch.zeros_like(gen_logits)
        context = torch.matmul(decode_attn, memory)
        copy_logits = copy_logits.scatter_add(2, src, decode_attn)
        prob = self.sig_proj(self.prob_proj(torch.cat([context, decode_output], -1)))

        gen_logits = prob * self.gen_softmax(gen_logits)
        copy_logits = (1 - prob) * self.copy_softmax(copy_logits)
        final_logits = gen_logits + copy_logits
        return torch.log(final_logits.squeeze(1).contiguous().view(-1, self.vocab_size))

class Bert(nn.Module):
    def __init__(self, large, temp_dir, model_pth=None, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained(model_pth, cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained(model_pth, cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, mask)
        return top_vec

class Roberta(nn.Module):
    def __init__(self, large, temp_dir, model_pth=None, finetune=False):
        super(Roberta, self).__init__()
        if(large):
            self.model = RobertaModel.from_pretrained(model_pth, cache_dir=temp_dir)
        else:
            self.model = RobertaModel.from_pretrained(model_pth, cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask)
        return top_vec

class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None, ids_to_tokens=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        if args.train_from == "bart":
            self.bert = RobertaModel(RobertaConfig("/home/lcl193798/PreRobertaSummMaro/src/config.json"))
        elif args.encoder == 'bert' or args.encoder == 'zh_bert':
            self.bert = Bert(args.large, args.temp_dir, args.model_pth, args.finetune_bert)
        elif args.encoder == 'roberta':
            self.bert = Roberta(args.large, args.temp_dir, args.model_pth, args.finetune_bert)



        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        if args.train_from == "bart":
            self.vocab_size = self.bert.config.vocab_size
        else:
            self.vocab_size = self.bert.model.config.vocab_size
        if args.encoder == 'roberta': 
            if args.train_from == "bart":
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.config.hidden_size, padding_idx=1)
            else:
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=1)
        else:
            tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
            
        if (self.args.share_emb) and self.args.train_from != 'bart':
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        self.decoder = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, train_from=self.args.train_from)

        '''
        else:
            args_bart = checkpoint['args_bart']
            tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=1)
            dictionary = [0]*self.vocab_size 
            self.decoder = TransformerBartDecoder(args_bart, dictionary, tgt_embeddings)
        '''
        if self.args.p_gen:
            self.generator = CopyGenerator(self.vocab_size, self.args.dec_hidden_size)
            #print (self.generator.gen_proj)
            self.generator.gen_proj.weight = self.decoder.embeddings.weight
        else:
            self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
            #print (self.generator)
            #print (self.generator[0])
            self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb) and args.train_from != 'bart':
                if args.encoder == "roberta":
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=1)
                else:
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
            if self.args.p_gen:
                self.generator.gen_proj.weight = self.decoder.embeddings.weight
            else:
                self.generator[0].weight = self.decoder.embeddings.weight
        if bert_from_extractive is not None:
            #print ([n for n, p in bert_from_extractive.items()])
            self.bert.model.load_state_dict(
                dict([(n[5:], p) for n, p in bert_from_extractive.items() if n.startswith('bert')]), strict=True)

        self.to(device)

    def forward(self, src, tgt, mask_src, mask_tgt):
        top_vec = self.bert(src, None, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, attns, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, attns[-1], top_vec, None

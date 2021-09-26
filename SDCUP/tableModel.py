# -*- encoding:utf-8 -*-

import torch.nn.functional as F
import torch.optim.lr_scheduler
import numpy as np
from uer.models.model import Model
from uer.model_builder import build_model
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu
import torch.nn as nn
from torch.autograd import Variable

from matplotlib.pylab import *

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda()
    return torch.autograd.Variable(x)

def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv

class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))

        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.data.copy_(torch.from_numpy(b))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False, U=None, V=None, ctx=None, l_hs=None):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape

    # sort before packking
    l = array(l)
    perm_idx = argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float()  # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        if ctx is None:
            # Take only final outputs for each columns.
            wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
            wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]
        else:
            ctx = ctx.unsqueeze(1)
            # [batch_size, 1, dim_emb] -> [batch_size, 1, hS]
            wenc_u = U(ctx)
            # [batch_size, seq_len, dim_emb] -> [batch_size, seq_len, hS]
            wenc_v = V(wenc)
            start = 0
            # [batch_size, 1, dim_emb]
            wenc2 = torch.zeros(wenc.shape[0], 1, wenc.shape[2])
            for b in range(ctx.shape[0]):
                # [1, hS] * [batch_size, seq_len, hS] -> [batch_size, seq_len, hS]
                attn = torch.mul(wenc_u[b], wenc_v[start:start + l_hs[b]])
                # attn, _ = nn.utils.rnn.pad_packed_sequence(attn, batch_first=True)
                # [batch_size, seq_len]
                attn = F.softmax(attn.sum(2), dim=1)
                wenc1 = torch.bmm(attn.unsqueeze(1), wenc[start:start + l_hs[b]])
                wenc1 += ctx[b]
                wenc2[start:start + l_hs[b]] = wenc1
                start += l_hs[b]
            wenc = wenc2

    wenc = wenc[perm_idx_inv]

    if return_hidden:
        # hout.shape = [number_of_directoin * num_of_layer, seq_len(=batch size), dim * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc



def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs, U=None, V=None, ctx=None):
    # wenc_hpu, hout, cout = encode(lstm,
    #                               wemb_hpu,
    #                               l_hpu,
    #                               return_hidden=True,
    #                               hc0=None,
    #                               last_only=True,
    #                               U=U,
    #                               V=V,
    #                               ctx=ctx,
    #                               l_hs=l_hs)
    # print("wemb_hpu:", wemb_hpu.shape)
    emb_hs_mean = torch.mean(wemb_hpu, dim=1)
    # print('mean:', emb_hs_mean.shape)
    wenc_hpu = emb_hs_mean
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)
    # print('l heasers:', l_hs)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    # print('w enc hs:', wenc_hs.shape)

    return wenc_hs


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class TableTextPretraining(nn.Module):
    def __init__(self, args):
        super(TableTextPretraining, self).__init__()

        # self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        # self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.config = args
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.pre_encoder = build_model(args)
        if args.use_cuda:
            pretrained_model = torch.load(args.pretrained_model_path)
        else:
            pretrained_model = torch.load(args.pretrained_model_path, map_location='cpu')
        print('loading model from table Text model:', args.pretrained_model_path)
        self.pre_encoder.load_state_dict(pretrained_model, strict=False)
        self.pre_encoder.to(args.device)

        # MLM.
        self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size)
        self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.device = args.device
        self.use_cuda = args.use_cuda
        hidden_size = 128  # int(args.iS / 2)

        self.lstm_hiddens = hidden_size
        self.lstm_layers = args.lS
        self.dropout_lstm_input = args.dr
        self.dropout_lstm_hidden = args.dr
        self.dropout_mlp = args.dr

        self.input_dims = 768
        self.enc_h = nn.LSTM(input_size=self.input_dims, hidden_size=int(self.lstm_hiddens / 2),
                             num_layers=self.lstm_layers, batch_first=True,
                             dropout=self.dropout_lstm_hidden, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=self.input_dims, hidden_size=int(self.lstm_hiddens / 2),
                             num_layers=self.lstm_layers, batch_first=True,
                             dropout=self.dropout_lstm_hidden, bidirectional=True)

        # Schema Dependency
        self.mlp_all = NonLinear(
            input_size=768,
            hidden_size=400,
            activation=nn.LeakyReLU(0.1))

        # self.mlp_arc_dep1 = NonLinear(
        #     input_size=400,
        #     hidden_size=args.mlp_arc_size + args.mlp_rel_size,
        #     activation=nn.LeakyReLU(0.1))
        # self.mlp_arc_head1 = NonLinear(
        #     input_size=400,
        #     hidden_size=args.mlp_arc_size + args.mlp_rel_size,
        #     activation=nn.LeakyReLU(0.1))

        # self.total_num = int((args.mlp_arc_size+args.mlp_rel_size) / 100)
        # self.arc_num = int(args.mlp_arc_size / 100)
        # self.rel_num = int(args.mlp_rel_size / 100)
        #
        # self.arc_biaffine1 = Biaffine(args.mlp_arc_size, args.mlp_arc_size, 1, bias=(True, False))
        # self.rel_biaffine1 = Biaffine(args.mlp_rel_size, args.mlp_rel_size, 9, bias=(True, True))
        #
        # # self.relation_weights = torch.FloatTensor(args.rel_weights).to(args.device)
        # self.auto_loss = AutomaticWeightedLoss(2)



if __name__ == '__main__':
    pass


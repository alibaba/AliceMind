import os, json
from copy import deepcopy
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sqlova.utils.utils import topk_multi_dim
from sqlova.utils.utils_wikisql import *


class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.max_wn = 6
        self.max_slen = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops

        self.question_knowledge_dim = 20
        self.header_knowledge_dim = 8

        self.enc = Encoder(iS, hS, dr)
        self.lstm = LSTM(iS, hS, lS, dr)
        self.slenp = SLENP(iS + hS, hS, lS, dr, self.max_slen)
        self.scp = SCP(iS, hS, lS, dr, self.question_knowledge_dim, self.header_knowledge_dim)
        self.ccop = CCO(iS + hS, hS, lS, dr)
        self.sap_multi = SAP_multi(iS  + hS + self.header_knowledge_dim, hS, lS, dr, n_agg_ops, self.max_slen)
        self.wnp = WNP(iS + hS, hS, lS, dr, self.max_wn)
        self.wcp = WCP(iS + hS + self.header_knowledge_dim, hS, lS, dr, self.max_wn)
        self.wop = WOP(iS + hS + self.header_knowledge_dim, hS, lS, dr, n_cond_ops, self.max_wn)
        self.wvp = WVP_se(iS + hS, hS, lS, dr, n_cond_ops, self.max_wn, self.question_knowledge_dim, self.header_knowledge_dim)  # start-end-search-discriminative model

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                g_cond_conn_op=None, g_slen=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False,
                knowledge = None, knowledge_header = None):
        # print(11)
        # sc
        ctx = wemb_n[:, 0, :]  # batch, hS(==iS)
        wenc_hs = self.enc(ctx, wemb_hpu, l_hpu, l_hs)  # batch, l_hs[b], hS
        # print('emb hpu', wemb_hpu.shape)
        # print("ctx: ", ctx.shape)
        # print("wemb_n: ", wemb_n.shape)
        # print("wenc_hs: ", wenc_hs.shape)
        # print('l n:', l_n)
        # print('l hpu:', l_hpu)
        # print('l hs:', l_hs)
        # print('***\n\n')

        lstm_n, lstm_ctx, lstm_h = self.lstm(wemb_n, l_n, wemb_hpu, l_hpu, l_hs)
        # print("lstm_n: ",  lstm_n.shape)
        # print("lstm_ctx: ", lstm_ctx.shape)
        # print("lstm_h: ", lstm_h.shape)

        mL_n = max(l_n)
        bS = len(l_hs)
        knowledge_use = [k + (mL_n - len(k)) * [0] for k in knowledge]
        knowledge_use = torch.tensor(knowledge_use).unsqueeze(-1)

        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1,
                                                                              index=knowledge_use,
                                                                              value=1).to(device)
        knowledge_header_use = [k + (max(l_hs) - len(k)) * [0] for k in knowledge_header]
        knowledge_header_use = torch.tensor(knowledge_header_use).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1,
                                                                                  index=knowledge_header_use,
                                                                                  value=1).to(device)

        ctx_all = torch.cat((ctx, lstm_ctx), 1)
        n_all = torch.cat((wemb_n, lstm_n, feature), 2)
        h_all = torch.cat((wenc_hs, lstm_h, feature2), 2)

        # print('lstm n:', lstm_n.shape)
        # print('lstm h:', lstm_h.shape)
        # print('lstm ctx:', lstm_ctx.shape)
        # print('feature:', feature.shape)
        # print('feature2:', feature2.shape)
        #
        #
        # print("ctx_all: ",  ctx_all.shape)
        # print("n_all: ", n_all.shape)
        # print("h_all: ", h_all.shape)

        s_slen = self.slenp(ctx_all)

        if g_slen:
            pr_slen = g_slen
        else:
            pr_slen = pred_slen(s_slen)

        # print(13)
        # s_sc = self.scp(wenc_hs, l_hs)
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc,
                             knowledge=knowledge, knowledge_header=knowledge_header)

        if g_sc:
            pr_sc = g_sc
        else:
            # pr_sc = pred_sc(s_sc)
            pr_sc = pred_sc_multi(pr_slen, s_sc)
        # print(14)
        # sa
        # s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
        s_sa = self.sap_multi(h_all, pr_slen, pr_sc)

        # print(15)
        s_cco = self.ccop(ctx_all)

        # print(16)
        # wn
        s_wn = self.wnp(ctx_all)
        #s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)
        # print(17)
        # wc
        s_wc = self.wcp(h_all, l_hs)
        # s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)
        # print(18)
        # wo
        s_wo = self.wop(h_all, pr_wn, pr_wc)

        s_wv = self.wvp(n_all, l_n, h_all, l_hs, wn=pr_wn, wc=pr_wc)

        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_cco, s_slen

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb,
                     nlu_t, nlu_wp_t, wp_to_wh_index, nlu,
                     beam_size=4,
                     show_p_sc=False, show_p_sa=False,
                     show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        """
        Execution-guided beam decoding.
        """
        ctx = wemb_n[:, 0, :]  # batch, hS(==iS)
        wenc_hs = self.enc(ctx, wemb_hpu, l_hpu, l_hs)  # batch, l_hs[b], hS

        s_slen = self.slenp(ctx)
        # sc
        s_sc = self.scp(wenc_hs, l_hs)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap_multi(wenc_hs, s_slen, pr_sc)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc]  # [B]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        # First flatten to 1-d
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.

        idxs = remap_sc_idx(idxs, pr_sc_beam)  # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:  # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break

        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # Now, Where-clause beam search.
        s_wn = self.wnp(ctx)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        s_wc = self.wcp(wenc_hs, l_hs)
        prob_wc = F.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.max_wn] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)  # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]

        # get most probable max_wn where-clouses
        # wo
        s_wo_max = self.wop(wenc_hs, pr_wn_max, pr_wc_max)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, max_wn, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.max_wn] * bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, wenc_hs, l_hs, wn=pr_wn_max, wc=pr_wc_max)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            # pr_wvi_beam = [B, max_wn, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, max_wn, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops - 1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops - 1):  # do not use final one
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.max_wn:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]],
                                                             [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        # s_wv = [B, max_wn, max_nlu_tokens, 2]
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i


class LayerNorm(nn.Module):
    def __init__(self, hS, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hS))
        self.beta = nn.Parameter(torch.zeros(hS))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        # x = input to the neuron.
        # normalize each vector (each token).
        # regularize x.
        # If x follows Gaussian distribution, it becomes standard Normal distribution (i.e., mu=0, std=1).
        u = x.mean(-1, keepdim=True)  # keepdim = keeprank of tensor.
        s = (x - u).pow(2).mean(-1, keepdim=True) # variance
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # standard
        x = x.to(device)
        # Gamma & Beta is trainable parameters.
        return self.gamma * x + self.beta


class Encoder(nn.Module):
    def __init__(self, iS=300, hS=100, dr=0.3):
        super(Encoder, self).__init__()
        self.iS = iS
        self.hS = hS
        self.dr = dr

        self.enc_h = nn.Linear(iS * 2, iS)
        self.layer_norm = LayerNorm(iS)
        self.U = nn.Linear(iS, iS)
        self.V = nn.Linear(iS, iS)

    def forward(self, ctx, wemb_hpu, l_hpu, l_hs):
        wenc = wemb_hpu
        ctx = ctx.unsqueeze(1)
        wenc_u = self.U(ctx)
        wenc_v = self.V(wenc)
        start = 0
        wenc2 = torch.zeros(wenc.shape[0], 1, wenc_v.shape[2])
        for b in range(ctx.shape[0]):
            attn = torch.mul(wenc_u[b], wenc_v[start:start + l_hs[b]])
            attn = F.softmax(attn.sum(2), dim=1)  # [batch_size, seq_len]
            wenc1 = torch.bmm(attn.unsqueeze(1), wenc[start:start + l_hs[b]])
            _bs = wenc1.shape[0]
            wenc1 = torch.cat([wenc1, ctx[b].unsqueeze(0).expand(_bs, -1, -1)], dim=-1)
            wenc1 = self.enc_h(wenc1)
            wenc2[start:start + l_hs[b]] = wenc1
            start += l_hs[b]

        wenc_hpu = wenc2.squeeze(1)
        hS = wenc_hpu.size(-1)

        wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
        wenc_hs = wenc_hs.to(device)

        # Re-pack according to batch.
        # ret = [B_NLq, max_len_headers_all, dim_lstm]
        st = 0
        for i, l_hs1 in enumerate(l_hs):
            wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
            st += l_hs1

        wenc_hs = self.layer_norm(wenc_hs)
        return wenc_hs

class LSTM(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(LSTM, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.U = nn.Linear(hS, hS)
        self.V = nn.Linear(hS, hS)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]
        ctx = wenc_n[:, 0, :]
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs, U=self.U, V=self.V, ctx=ctx)  # [b, hs, dim]
        return wenc_n, ctx, wenc_hs


class SCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, question_knowledge_dim=15, header_knowledge_dim=4):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.question_knowledge_dim = question_knowledge_dim
        self.header_knowledge_dim = header_knowledge_dim

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.U = nn.Linear(hS, hS)
        self.V = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS + self.header_knowledge_dim, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False,
                  knowledge=None, knowledge_header=None):
        
        mL_n = max(l_n)
        bS = len(l_hs)
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]
        ctx = wenc_n[:, 0, :]
        knowledge = [k + (mL_n - len(k)) * [0] for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)

        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1,
                                                                              index=knowledge,
                                                                              value=1).to(device)

        wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs, U=self.U, V=self.V, ctx=ctx)  # [b, hs, dim]
        # print(wenc_hs.shape)
        knowledge_header = [k + (max(l_hs) - len(k)) * [0] for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1,
                                                                                        index=knowledge_header,
                                                                                        value=1).to(device)

        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)
        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))
        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            # p = [b, hs, n]
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig = figure(2001, figsize=(12, 3.5))
            # subplot(6,2,7)
            subplot2grid((7, 2), (3, 0), rowspan=2)
            cla()
            _color = 'rgbkcm'
            _symbol = '.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--' + _symbol[color_idx] + _color[color_idx], ms=7)

            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2)  # [bS, mL_hs, 1] -> [bS, mL_hs]

        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000

        return s_sc

# class SCP(nn.Module):
#     def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
#         super(SCP, self).__init__()
#         self.iS = iS
#         self.hS = hS
#         self.lS = lS
#         self.dr = dr
# 
#         self.sc_out = nn.Sequential(nn.Linear(iS, hS),
#                                     nn.Dropout(dr),
#                                     nn.Tanh(),
#                                     nn.Linear(hS, 1))
# 
#     def forward(self, wenc_hs, l_hs):
# 
#         s_sc = self.sc_out(wenc_hs).squeeze(2)  # [bS, mL_hs, 1] -> [bS, mL_hs]
# 
#         # Penalty
#         mL_hs = max(l_hs)
#         for b, l_hs1 in enumerate(l_hs):
#             if l_hs1 < mL_hs:
#                 s_sc[b, l_hs1:] = -1e+10
#         return s_sc


class SLENP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, mL_s=3):
        super(SLENP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_s = mL_s  # max select number

        self.wn_out = nn.Sequential(nn.Linear(iS, hS),
                                    nn.Dropout(dr),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_s + 1))  # max number (3 + 1)

    def forward(self, ctx):
        s_wn = self.wn_out(ctx)

        return s_wn


class CCO(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(CCO, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.sc_out = nn.Sequential(nn.Linear(iS, hS),
                                    nn.Dropout(dr),
                                    nn.Tanh(),
                                    nn.Linear(hS, 3))

    def forward(self, ctx):
        s_sc = self.sc_out(ctx)
        return s_sc


class SAP_multi(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=3, mL_s=3):
        super(SAP_multi, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_s = mL_s  # max select num
        self.wo_out = nn.Sequential(
            nn.Linear(iS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_cond_ops)
        )

    def forward(self, wenc_hs, sn, sc):

        bS = wenc_hs.shape[0]
        # wn

        wenc_hs_ob = []  # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            # print(type([wenc_hs[b, col] for col in wc[b]]))
            real = [wenc_hs[b, col] for col in sc[b]]
            pad = (self.mL_s - sn[b]) * [
                torch.zeros_like(wenc_hs[b, 0])]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        s_wo = self.wo_out(wenc_hs_ob)

        return s_wo


# class WNP(nn.Module):
#     def __init__(self, iS=300, hS=100, lS=2, dr=0.3, ):
#         super(WNP, self).__init__()
#         self.iS = iS
#         self.hS = hS
#         self.lS = lS
#         self.dr = dr
# 
#         self.mL_w = 4  # max where condition number
# 
#         self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
#                              num_layers=lS, batch_first=True,
#                              dropout=dr, bidirectional=True)
# 
#         self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
#                              num_layers=lS, batch_first=True,
#                              dropout=dr, bidirectional=True)
# 
#         self.W_att_h = nn.Linear(hS, 1)
#         self.W_hidden = nn.Linear(hS, lS * hS)
#         self.W_cell = nn.Linear(hS, lS * hS)
# 
#         self.W_att_n = nn.Linear(hS, 1)
#         self.wn_out = nn.Sequential(nn.Linear(hS, hS),
#                                     nn.Tanh(),
#                                     nn.Linear(hS, self.mL_w + 1))  # max number (4 + 1)
# 
#         self.softmax_dim1 = nn.Softmax(dim=1)
#         self.softmax_dim2 = nn.Softmax(dim=2)
# 
#     def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=False):
#         # Encode
# 
#         wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]
# 
#         bS = len(l_hs)
#         mL_n = max(l_n)
#         mL_hs = max(l_hs)
#         # mL_h = max(l_hpu)
# 
#         #   (self-attention?) column Embedding?
#         #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs]
#         att_h = self.W_att_h(wenc_hs).squeeze(2)
# 
#         #   Penalty
#         for b, l_hs1 in enumerate(l_hs):
#             if l_hs1 < mL_hs:
#                 att_h[b, l_hs1:] = -10000000000
#         p_h = self.softmax_dim1(att_h)
# 
#         if show_p_wn:
#             if p_h.shape[0] != 1:
#                 raise Exception("Batch size should be 1.")
#             fig = figure(2001);
#             subplot(7, 2, 5)
#             cla()
#             plot(p_h[0].data.numpy(), '--rs', ms=7)
#             title('wn: header_weight')
#             grid(True)
#             fig.canvas.draw()
#             show()
#             # input('Type Eenter to continue.')
# 
#         #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
#         c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)
# 
#         #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
#         hidden = self.W_hidden(c_hs)  # [B, 4, 200/2]
#         hidden = hidden.view(bS, self.lS * 2, int(
#             self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
#         hidden = hidden.transpose(0, 1).contiguous()
# 
#         cell = self.W_cell(c_hs)  # [B, 4, 100/2]
#         cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2]
#         cell = cell.transpose(0, 1).contiguous()
# 
#         wenc_n = encode(self.enc_n, wemb_n, l_n,
#                         return_hidden=False,
#                         hc0=(hidden, cell),
#                         last_only=False)  # [b, n, dim]
# 
#         att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]
# 
#         #    Penalty
#         for b, l_n1 in enumerate(l_n):
#             if l_n1 < mL_n:
#                 att_n[b, l_n1:] = -10000000000
#         p_n = self.softmax_dim1(att_n)
# 
#         #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
#         c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
#         s_wn = self.wn_out(c_n)
# 
#         return s_wn


class WNP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, mL_w=4):
        super(WNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = mL_w  # max where condition number

        self.wn_out = nn.Sequential(nn.Linear(iS, hS),
                                    nn.Dropout(dr),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_w + 1))  # max number (4 + 1)

    def forward(self, ctx):
        s_wn = self.wn_out(ctx)

        return s_wn


# class WCP(nn.Module):
#     def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
#         super(WCP, self).__init__()
#         self.iS = iS
#         self.hS = hS
#         self.lS = lS
#         self.dr = dr
# 
#         self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
#                              num_layers=lS, batch_first=True,
#                              dropout=dr, bidirectional=True)
# 
#         self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
#                              num_layers=lS, batch_first=True,
#                              dropout=dr, bidirectional=True)
# 
#         self.W_att = nn.Linear(hS, hS)
#         self.U = nn.Linear(hS, hS)
#         self.V = nn.Linear(hS, hS)
#         self.W_c = nn.Linear(hS, hS)
#         self.W_hs = nn.Linear(hS, hS)
#         # self.W_out = nn.Sequential(
#         #     nn.Tanh(), nn.Linear(2 * hS, 1)
#         # )
#         # maybe tough
#         self.W_out = nn.Sequential(
#             nn.Linear(2 * hS, hS),
#             nn.Tanh(),
#             nn.Linear(hS, 4)
#         )
# 
#         self.softmax_dim1 = nn.Softmax(dim=1)
#         self.softmax_dim2 = nn.Softmax(dim=2)
# 
#     def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc, penalty=True):
#         # Encode
#         wenc_n = encode(self.enc_n, wemb_n, l_n,
#                         return_hidden=False,
#                         hc0=None,
#                         last_only=False)  # [b, n, dim]
# 
#         ctx = wenc_n[:, 0, :]
#         wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs, U=self.U, V=self.V, ctx=ctx)  # [b, hs, dim]
# 
#         # attention
#         # wenc = [bS, mL, hS]
#         # att = [bS, mL_hs, mL_n]
#         # att[b, i_h, j_n] = p(j_n| i_h)
#         att = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))
# 
#         # penalty to blank part.
#         mL_n = max(l_n)
#         for b_n, l_n1 in enumerate(l_n):
#             if l_n1 < mL_n:
#                 att[b_n, :, l_n1:] = -10000000000
# 
#         # make p(j_n | i_h)
#         p = self.softmax_dim2(att)
# 
#         # max nlu context vectors
#         # [bS, mL_hs, mL_n]*[bS, mL_hs, mL_n]
#         wenc_n = wenc_n.unsqueeze(1)  # [ b, n, dim] -> [b, 1, n, dim]
#         p = p.unsqueeze(3)  # [b, hs, n] -> [b, hs, n, 1]
#         c_n = torch.mul(wenc_n, p).sum(2)  # -> [b, hs, dim], c_n for each header.
# 
#         y = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)  # [b, hs, 2*dim]
#         score = self.W_out(y)  # [b, hs, 4]
# 
#         if penalty:
#             for b, l_hs1 in enumerate(l_hs):
#                 score[b, l_hs1:] = -1e+10
# 
#         return score

class WCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, mL_w=4):
        super(WCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = mL_w  # max where condition number

        self.W_out = nn.Sequential(
            nn.Linear(iS, hS),
            nn.Dropout(dr),
            nn.Tanh(),
            nn.Linear(hS, mL_w)
        )

    def forward(self, wenc_hs, l_hs):

        score = self.W_out(wenc_hs)  # [b, ml_h, 4]
        for b, l_hs1 in enumerate(l_hs):
            score[b, l_hs1:] = -1e+10

        return score


class WOP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=3, mL_w=4):
        super(WOP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = mL_w  # max where condition number

        self.repeat = nn.Parameter(torch.zeros((self.mL_w, self.iS)))

        self.wo_out = nn.Sequential(
            nn.Linear(iS, hS),
            nn.Dropout(dr),
            nn.Tanh(),
            nn.Linear(hS, n_cond_ops)
        )

    def forward(self, wenc_hs, wn, wc):

        bS = wenc_hs.shape[0]

        wenc_hs_ob = []  # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            # real = [wenc_hs[b, col] for col in wc[b]]
            # pad = (self.mL_w - wn[b]) * [
            #     torch.zeros_like(wenc_hs[b, 0])]  # this padding could be wrong. Test with zero padding later.
            real = []
            for col, val in enumerate(wc[b]):
                rep = 0
                for wc1 in range(0, col, 1):
                    if val == wc[b][wc1]:
                        rep += 1
                if rep==0:
                    real.append(wenc_hs[b, col])
                else:
                    real.append(wenc_hs[b, col] + self.repeat[rep])
            pad = (self.mL_w - wn[b]) * [torch.zeros(self.iS, device=device)]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        s_wo = self.wo_out(wenc_hs_ob)  # bS, mL_w, 3

        return s_wo


class WVP_se(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4,
                     mL_w=4, question_knowledge_dim=15, header_knowledge_dim=4):
        super(WVP_se, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops
        self.question_knowledge_dim = question_knowledge_dim
        self.header_knowledge_dim = header_knowledge_dim

        self.mL_w = mL_w  # max where condition number

        self.U1 = nn.Sequential(
            nn.Linear(iS + self.question_knowledge_dim, hS + self.question_knowledge_dim),
            nn.Dropout(dr)
        )
        self.V1 = nn.Sequential(
            nn.Linear(iS + self.question_knowledge_dim, hS + self.question_knowledge_dim),
            nn.Dropout(dr)
        )

        self.wv_s = nn.Sequential(
            nn.Linear(3 * iS + self.header_knowledge_dim + 2*self.question_knowledge_dim, hS),
            nn.Dropout(dr),
            nn.Tanh(),
            nn.Linear(hS, 1)
        )
        self.wv_e = nn.Sequential(
            nn.Linear(3 * iS + self.header_knowledge_dim + 2*self.question_knowledge_dim, hS),
            nn.Dropout(dr),
            nn.Tanh(),
            nn.Linear(hS, 1)
        )

    def forward(self, wenc_n, l_n, wenc_hs, l_hs, wn, wc):

        bS = len(l_hs)
        ctx = wenc_n[:, 0, :]
        s_wv = torch.zeros(bS, self.mL_w, wenc_n.shape[1], 2)
        s_wv = s_wv.to(device)

        for idx in range(self.mL_w):
            wenc_hs_ob = []
            for b in range(bS):
                emb_c = torch.zeros_like(ctx[0])
                # the two columns need to be the same
                if idx != 0:
                    start = s_wv[b, idx - 1, :, 0].argmax(dim=0)
                    end = s_wv[b, idx - 1, start:, 1].argmax(dim=0) + start
                    # [1, 1, hS]
                    ctx_g = self.U1(ctx[b].unsqueeze(0).unsqueeze(1))
                    # [1, seq_len, hS]
                    ctx_v = self.V1(wenc_n[b, start: end + 1, :].unsqueeze(0))
                    attn = torch.mul(ctx_g, ctx_v)
                    attn = F.softmax(attn.sum(2), dim=1)
                    emb_c = torch.matmul(attn.squeeze(0), wenc_n[b, start: end + 1, :])

                if len(wc[b]) > idx:
                    emb = [torch.cat((wenc_hs[b, wc[b][idx]], emb_c), dim=0)]
                else:
                    emb = [torch.cat((torch.zeros_like(wenc_hs[b, 0]), torch.zeros_like(ctx[0])), dim=0)]
                wenc_hs_ob1 = torch.stack(emb)
                wenc_hs_ob.append(wenc_hs_ob1)

            # list to [B, 1, dim] tensor.
            wenc_hs_ob = torch.stack(wenc_hs_ob)
            wenc_hs_ob = wenc_hs_ob.to(device)
            # wenc_n: [bS, mL_n, dim]
            wenc_ne = wenc_n.unsqueeze(1)  # [bS, 1, mL, dim]
            mL_n = max(l_n)
            wenc_hs_ob_ne = wenc_hs_ob.unsqueeze(2).expand(-1, -1, mL_n, -1)

            vec2 = torch.cat([wenc_hs_ob_ne, wenc_ne], dim=3)  # [bS, 1, mL, 3 * dim]

            s_wv[:, idx, :, 0] = self.wv_s(vec2).squeeze()  # [bS, mL]
            s_wv[:, idx, :, 1] = self.wv_e(vec2).squeeze()

            # penalty for spurious tokens
            for b, l_n1 in enumerate(l_n):
                if l_n1 < mL_n:
                    s_wv[b, idx, l_n1:, :] = -10000000000

        return s_wv


def Loss_sw_se(s_sc, s_cco, s_sa, s_wn, s_wc, s_wo, s_wv, s_slen, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, g_cond_conn_op,
               g_slen):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_slen(s_slen, g_slen)
    # loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sc_multi(s_sc, g_sc)
    loss += Loss_scco(s_cco, g_cond_conn_op)
    # loss += Loss_sa(s_sa, g_sa)
    loss += Loss_sa_multi(s_sa, g_slen, g_sa)
    loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv, g_wn, g_wvi)

    return loss, Loss_slen(s_slen, g_slen), Loss_sc_multi(s_sc, g_sc), Loss_scco(s_cco, g_cond_conn_op), \
           Loss_sa_multi(s_sa, g_slen, g_sa), Loss_wn(s_wn, g_wn), Loss_wc(s_wc, g_wc), Loss_wo(s_wo, g_wn, g_wo), \
           Loss_wv_se(s_wv, g_wn, g_wvi)


def Loss_slen(s_slen, g_slen):
    loss = F.cross_entropy(s_slen, torch.tensor(g_slen).to(device))

    return loss


def Loss_sc(s_sc, g_sc):
    loss = F.cross_entropy(s_sc, torch.tensor(g_sc).to(device))
    return loss


def Loss_sc_multi(s_wc, g_wc):
    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0 / len(g_wc1)
    p = F.log_softmax(s_wc, dim=1)
    loss = F.kl_div(p, im)
    return loss


def Loss_sa(s_sa, g_sa):
    loss = F.cross_entropy(s_sa, torch.tensor(g_sa).to(device))
    return loss


def Loss_scco(s_cco, g_cond_conn_op):
    loss = F.cross_entropy(s_cco, torch.tensor(g_cond_conn_op).to(device))
    return loss


def Loss_sa_multi(s_sa, g_sslen, g_sa):
    loss = 0
    for b, g_sslen1 in enumerate(g_sslen):
        if g_sslen1 == 0:
            continue
        g_sa1 = g_sa[b]
        s_sa1 = s_sa[b]
        g_sa1 = [int(x) for x in g_sa1]
        loss += F.cross_entropy(s_sa1[:g_sslen1], torch.autograd.Variable(torch.tensor(g_sa1).to(device)))
    return loss


def Loss_wn(s_wn, g_wn):
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))

    return loss



def Loss_wc(s_wc, g_wc):
    # Construct index matrix
    # [bS, max_h_len, 4]
    bS, max_h_len, dim = s_wc.shape

    loss = 0
    for b, g_wc1 in enumerate(g_wc):
        l = len(g_wc1)
        s_wc_l = s_wc[b, :, :l]
        loss += F.cross_entropy(s_wc_l.permute(1, 0), torch.autograd.Variable(torch.tensor(g_wc1).to(device)))

    return loss


def Loss_wo(s_wo, g_wn, g_wo):
    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.autograd.Variable(torch.tensor(g_wo1).to(device)))

    return loss


def Loss_wv_se(s_wv, g_wn, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0

    # g_wvi = torch.tensor(g_wvi).to(device)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:, 0]
        g_ed1 = g_wvi1[:, 1]
        # loss from the start position
        # print("st_login: ", s_wv[b,:g_wn1,:,0].shape, g_st1)
        loss += F.cross_entropy(s_wv[b, :g_wn1, :, 0], g_st1)

        # loss from the end position
        # print("ed_login: ", s_wv[b,:g_wn1,:,1].shape, g_ed1)
        loss += F.cross_entropy(s_wv[b, :g_wn1, :, 1], g_ed1)

    return loss


# ========= Decoder-Layer ===========
class FT_s2s_1(nn.Module):
    """ Decoder-Layer """

    def __init__(self, iS, hS, lS, dr, max_seq_length, n_cond_ops, n_agg_ops, old=False):
        super(FT_s2s_1, self).__init__()
        self.iS = iS  # input_size
        self.hS = hS  # hidden_size
        self.ls = lS
        self.dr = dr

        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4

        self.decoder_s2s = Decoder_s2s(iS, hS, lS, dr, max_seq_length)

    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None):
        score = self.decoder_s2s(wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs)
        return score

    def EG_forward(self, wenc_s2s, l_input, cls_vec,
                   pnt_start_tok, pnt_end_tok,
                   i_sql_vocab, i_nlu, i_hds,  # for EG
                   tokens, nlu, nlu_t, hds, tt_to_t_idx,  # for EG
                   tb, engine,
                   beam_size=4, beam_only=True):
        """ EG-guided beam-search """

        score = self.decoder_s2s.EG_forward(wenc_s2s, l_input, cls_vec,
                                            pnt_start_tok, pnt_end_tok,
                                            i_sql_vocab, i_nlu, i_hds,  # for EG
                                            tokens, nlu, nlu_t, hds, tt_to_t_idx,  # for EG
                                            tb, engine,
                                            beam_size, beam_only)
        return score


class Decoder_s2s(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, max_seq_length=222, n_cond_ops=3):
        super(Decoder_s2s, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.mL = max_seq_length

        self.Tmax = 200

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.decode_pn = nn.LSTM(input_size=max_seq_length, hidden_size=hS,
                                 num_layers=lS, batch_first=True,
                                 dropout=dr)

        self.W_s2s = nn.Linear(iS, hS)
        self.W_pnt = nn.Linear(hS, hS)

        self.wv_out = nn.Sequential(nn.Tanh(), nn.Linear(hS, 1))

    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None, ):

        # Encode
        bS, mL_input, iS = wenc_s2s.shape

        # Now, pointer network.
        ipnt = wenc_s2s.new_zeros(bS, 1, mL_input).to(device)  # [B, 1, 200]
        ipnt[:, 0, pnt_start_tok] = 1  # 27 is of start token under current tokenization scheme

        # initial (current) pointer
        cpnt = ipnt

        # reshape wenc_s2s to incorporate T later
        wenc_s2s = wenc_s2s.unsqueeze(1)
        # h_0 and c_0 from cls_vec
        # They are not bidirectional.
        h_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        c_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        for i_layer in range(self.lS):
            h_st = (2 * i_layer) * self.hS
            h_ed = h_st + self.hS

            c_st = (2 * i_layer + 1) * self.hS
            c_ed = c_st + self.hS

            h_0[i_layer] = cls_vec[:, h_st:h_ed]  # [ # of layers, batch, dim]
            c_0[i_layer] = cls_vec[:, c_st:c_ed]  # [ # of layers, batch, dim]

        if g_pnt_idxs:

            pnt_n = torch.zeros(bS, self.Tmax, mL_input).to(device)  # one hot
            # assign index
            for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
                for t, g_pnt_idx in enumerate(g_pnt_idxs1):
                    pnt_n[b, t, g_pnt_idx] = 1

            # Encode
            dec_pn, _ = self.decode_pn(pnt_n, (h_0, c_0))
            dec_pn = dec_pn.contiguous()

            # [bS, T, iS]
            dec_pn = dec_pn.unsqueeze(2)

            # Calculate score
            s_wv = self.wv_out(
                self.W_s2s(wenc_s2s)
                + self.W_pnt(dec_pn)
            ).squeeze(3)  # [B, T, mL_input, dim] -> [B, T, mL_input, 1] -> [B, T, mL_input]
            # s_wv = [B, 4, T, mL_n] = [batch, conds, token idx, score]

            # penalty
            for b, l_input1 in enumerate(l_input):
                if l_input1 < mL_input:
                    s_wv[b, :, l_input1:] = -10000000000

        else:
            t = 0
            s_wv_list = []
            cpnt_h = (h_0, c_0)
            while t < self.Tmax:
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)  # lstm

                # [B, 1, 100] -> [B, 1, 1, 100]
                dec_pn = dec_pn.unsqueeze(2)
                # [bS, T, iS]

                # get score
                s_wv1 = self.wv_out(
                    self.W_s2s(wenc_s2s)  # [B, 1,   mL_input, dim]
                    + self.W_pnt(dec_pn)  # [B, T=1,        1, dim]   Now, T=1
                ).squeeze(3)
                # s_wv = [B, 4, 1, mL_n, 1] = [batch, conds, token idx, score]
                # -> [B, 4, mL_n]

                # Masking --
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000

                # Collect score--
                s_wv_list.append(s_wv1)

                # [B, 1, mL_input] -> [B, mL_n] -> [bS*(5-1)]
                # (max_val, max_indices)
                _val, pnt_n = s_wv1.view(bS, -1).max(dim=1)

                # formatting pnt_n as a one-hot input.
                cpnt = torch.zeros(bS, mL_input).to(device)
                # cpnt = cpnt.scatter_(dim=1, index=pnt_n.unsqueeze(1), src=1).to(device)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)

                cpnt = cpnt.unsqueeze(1)  # --> [B * 4, 1, 200]
                t += 1

            s_wv = torch.stack(s_wv_list, 1)  # [B,
            s_wv = s_wv.squeeze(2)  #
            # # Following lines seems to be unnecessary.
            # # Penalty to blank parts
            # for b, l_input1 in enumerate(l_input):
            #     if l_input1 < mL_input:
            #         s_wv[b, :, l_input1:] = -10000000000

        return s_wv

    def EG_forward(self, wenc_s2s, l_input, cls_vec,
                   pnt_start_tok, pnt_end_tok,
                   i_sql_vocab, i_nlu, i_hds,  # for EG
                   tokens, nlu, nlu_t, hds, tt_to_t_idx,  # for EG
                   tb, engine,
                   beam_size, beam_only=True):

        # Encode
        bS, mL_input, iS = wenc_s2s.shape

        # reshape wenc_s2s to incorperate T later
        wenc_s2s = wenc_s2s.unsqueeze(1)
        # h_0 and c_0 from cls_vec
        # They are not bidirectional.
        h_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        c_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        for i_layer in range(self.lS):
            h_st = (2 * i_layer) * self.hS
            h_ed = h_st + self.hS

            c_st = (2 * i_layer + 1) * self.hS
            c_ed = c_st + self.hS

            h_0[i_layer] = cls_vec[:, h_st:h_ed]  # [ # of layers, batch, dim]
            c_0[i_layer] = cls_vec[:, c_st:c_ed]  # [ # of layers, batch, dim]

        # initial (current) pointer
        pnt_list_beam = []
        cpnt_beam = []
        cpnt_h_beam = []

        for i_beam in range(beam_size):
            pnt_list_beam1 = []
            for b in range(bS):
                pnt_list_beam1.append([[pnt_start_tok], 0])
            pnt_list_beam.append(pnt_list_beam1)
            # initisl cpnt
            # Now, initialize pointer network.
            ipnt = wenc_s2s.new_zeros(bS, 1, mL_input).to(device)  # [B, 1, 200]
            # Distort ipnt by i_bam on purpose to avoid initial duplication of beam-search
            ipnt[:, 0, pnt_start_tok] = 1  # 27 is of start token under current tokenization scheme

            cpnt_beam.append(ipnt)
            cpnt_h_beam.append((h_0, c_0))
        t = 0
        while t < self.Tmax:
            # s_wv1_beam = []
            candidates = [[] for b in range(bS)]  # [bS]

            # Generate beam
            for i_beam, cpnt in enumerate(cpnt_beam):
                cpnt_h = cpnt_h_beam[i_beam]

                pnt_list_beam1 = pnt_list_beam[i_beam]
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)  # lstm
                cpnt_h_beam[i_beam] = cpnt_h

                # [B, 1, 100] -> [B, 1, 1, 100]
                dec_pn = dec_pn.unsqueeze(2)
                # [bS, T, iS]

                # get score
                s_wv1 = self.wv_out(
                    self.W_s2s(wenc_s2s)  # [B, 1,   mL_input, dim]
                    + self.W_pnt(dec_pn)  # [B, T=1,        1, dim]   Now, T=1
                ).squeeze(3)
                # s_wv = [B, 4, 1, mL_n, 1] = [batch, conds, token idx, score]
                # -> [B, 4, mL_n]

                # Masking --
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000

                # Get the candidates only among the input space.
                prob, idxs = F.softmax(s_wv1.view(bS, -1), dim=1).topk(dim=1, k=max(l_input))
                log_prob = torch.log(prob)  # [bS, beam_size]

                for b, log_prob1 in enumerate(log_prob):
                    pnt_list11, score = pnt_list_beam1[b]
                    for i_can, log_prob11 in enumerate(log_prob1):
                        # no update if last token was the end-token
                        previous_pnt = pnt_list11[-1]
                        if previous_pnt == pnt_end_tok:
                            new_seq = pnt_list11
                            new_score = score
                        else:
                            new_seq = pnt_list11 + [idxs[b][i_can].item()]
                            new_score = score + log_prob11.item()
                        _candidate = [new_seq, new_score]

                        candidates[b].append(_candidate)

            # Execution-guided beam filtering
            for b, candidates1 in enumerate(candidates):
                new_pnt_list_batch1 = sorted(candidates1, key=lambda list1: list1[-1], reverse=True)
                cnt = 0
                selected_candidates1 = []
                for new_pnt_list_batch11 in new_pnt_list_batch1:
                    if new_pnt_list_batch11 not in selected_candidates1:
                        if beam_only:
                            selected_candidates1.append(new_pnt_list_batch11)
                            pnt_list_beam[cnt][b] = new_pnt_list_batch11
                            cnt += 1
                        else:
                            # Need to be modified here.
                            executable = False
                            testable = False

                            pr_i_vg_list, pr_i_vg_sub_list = gen_i_vg_from_pnt_idxs([new_pnt_list_batch11[0]],
                                                                                    [i_sql_vocab[b]], [i_nlu[b]],
                                                                                    [i_hds[b]])
                            pr_sql_q_s2s, pr_sql_i = gen_sql_q_from_i_vg([tokens[b]], [nlu[b]], [nlu_t[b]], [hds[b]],
                                                                         [tt_to_t_idx[b]],
                                                                         pnt_start_tok, pnt_end_tok,
                                                                         [new_pnt_list_batch11[0]], pr_i_vg_list,
                                                                         pr_i_vg_sub_list)

                            # check testability from select-clause
                            try:
                                # check whether basic elements presents in pr_sql_i
                                # If so, it is testable.

                                idx_agg = pr_sql_i[0]["agg"]
                                idx_sel = pr_sql_i[0]["sel"]
                                testable = True
                            except:
                                testable = False
                                pass

                            # check the presence of conds
                            if testable:
                                try:
                                    conds = pr_sql_i[0]["conds"]
                                except:
                                    conds = []

                                try:
                                    pr_ans1 = engine.execute(tb[b]['id'], idx_sel, idx_agg, conds)
                                    executable = bool(pr_ans1)
                                except:
                                    executable = False

                            #
                            if testable:
                                if executable:
                                    add_candidate = True
                                else:
                                    add_candidate = False
                            else:
                                add_candidate = True

                            if add_candidate:
                                selected_candidates1.append(new_pnt_list_batch11)
                                pnt_list_beam[cnt][b] = new_pnt_list_batch11
                                cnt += 1

                    if cnt == beam_size:
                        break

                if cnt < beam_size:
                    # not executable at all..
                    # add junk sequence.
                    for i_junk in range(cnt, beam_size):
                        pnt_list_beam[i_junk][b] = [[pnt_end_tok], -9999999]

            # generate cpnt
            # formatting pnt_n as a one-hot input.
            for i_beam in range(beam_size):
                cpnt = torch.zeros(bS, mL_input).to(device)
                # cpnt = cpnt.scatter_(dim=1, index=pnt_n.unsqueeze(1), src=1).to(device)
                idx_batch = [seq_score[0][-1] for seq_score in pnt_list_beam[i_beam]]
                pnt_n = torch.tensor(idx_batch).to(device)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)
                cpnt = cpnt.unsqueeze(1)  # --> [B, t=1, mL_input]
                cpnt_beam[i_beam] = cpnt
            t += 1

        # Generate best pr_pnt_list, p_tot
        pr_pnt_idxs = []
        p_list = []
        for b in range(bS):
            pnt_list_beam_best = pnt_list_beam[0]
            pr_pnt_idxs.append(pnt_list_beam_best[b][0])
            p_list.append(pnt_list_beam_best[b][1])

        return pr_pnt_idxs, p_list, pnt_list_beam


# =============  Shallow-Layer ===============
class FT_Scalar_1(nn.Module):
    """ Shallow-Layer """

    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(FT_Scalar_1, self).__init__()
        self.iS = iS  # input_size
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4

    def scp(self, wemb_h, l_hs):
        bS, max_header_len, _ = wemb_h.shape
        # s_sc

        s_sc = torch.zeros(bS, max_header_len).to(device)
        s_sc[:, :] = wemb_h[:, :, 0]  # s_sc = [B, max_header length, 1]

        # s_sc[:,:] = F.tanh(wemb_h[:,:,0])  # s_sc = [B, max_header length, 1]
        # s_sc = s_sc.squeeze(2)
        # masking
        # print(f"s_sc {s_sc}")
        for b, l_hs1 in enumerate(l_hs):
            s_sc[b, l_hs1:] = -9999999999.0

        return s_sc

    def sap(self, wemb_h, pr_sc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape
        # select of aggregation operator
        s_sa = torch.zeros([bS, self.n_agg_ops]).to(device)
        for b, pr_sc1 in enumerate(pr_sc):
            s_sa[b, :] = wemb_h[b, pr_sc1, idx_st:idx_ed]

        return s_sa

    def wnp(self, cls_vec):
        bS = cls_vec.shape[0]
        # [B,hS] -> [B, n_where_num+1]
        s_wn = torch.zeros(bS, (self.n_where_num + 1)).to(device)
        s_wn[:, :] = cls_vec[:, 0:(self.n_where_num + 1)]

        return s_wn

    def wcp(self, wemb_h, l_hs, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape

        s_wc = torch.zeros(bS, max_header_len, 1).to(device)
        s_wc[:, :, :] = wemb_h[:, :, idx_st:idx_ed]

        s_wc = s_wc.squeeze(2)  # [B, max_header_length]

        # masking
        for b, l_hs1 in enumerate(l_hs):
            s_wc[b, l_hs1:] = -99999999999.0

        return s_wc

    def wop(self, wemb_h, pr_wc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape

        s_wo = torch.zeros([bS, self.n_where_num, self.n_cond_ops]).to(device)
        for b, pr_wc1 in enumerate(pr_wc):
            if len(pr_wc1) > 0:
                s_wo[b, 0:len(pr_wc1), :] = wemb_h[b, pr_wc1, idx_st:idx_ed]
            else:
                pass

        return s_wo

    def wvp(self, wemb_n, l_n, pr_wc):
        bS, _, _ = wemb_n.shape

        s_wv = torch.zeros([bS, self.n_where_num, max(l_n), 2]).to(device)
        for b, pr_wc1 in enumerate(pr_wc):

            if len(pr_wc1) > 0:
                # start logit
                s_wv[b, 0:len(pr_wc1), :, 0] = wemb_n[b, :, pr_wc1].transpose(0, 1)
                # end logit
                s_wv[b, 0:len(pr_wc1), :, 1] = wemb_n[b, :, [pr_wc11 + 100 for pr_wc11 in pr_wc1]].transpose(0, 1)
            else:
                pass

        # masking
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_n):
            if l_n1 < max(l_n):
                s_wv[b, :, l_n1:, :] = -1e+11
        return s_wv

    def forward(self, wemb_n, l_n, wemb_h, l_hs, cls_vec,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        # wemb_n = [B, max_nlu_token_length, hS] # here, # of target_layer is fixed to 1.
        # wemb_h = [B, max_header #, hS]

        s_sc = self.scp(wemb_h, l_hs)
        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # s_sa
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)

        if g_sa:
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)

        # where_number

        s_wn = self.wnp(cls_vec)
        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc
        idx_st = idx_ed + 1
        idx_ed = idx_st + 1
        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # wo
        idx_st = idx_ed + 1
        idx_ed = idx_st + self.n_cond_ops

        s_wo = self.wop(wemb_h, pr_wc, idx_st, idx_ed)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        # s_wv =  [bS, 4, mL, 2]
        s_wv = self.wvp(wemb_n, l_n, pr_wc)

        # print(s_wv)
        # s_wv = F.tanh(s_wv)
        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def forward_EG(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                   nlu_t, nlu_tt, tt_to_t_idx, nlu,
                   beam_size=4):
        """
        Execution-guided beam decoding.
        Essentially identical with that of NL2SQL Layer.
        """
        # Select-clause
        prob_sca, pr_sc_best, pr_sa_best, \
        p_sc_best, p_sa_best, p_select \
            = self.EG_decoding_select(wemb_h, l_hs, tb, beam_size=beam_size)

        # Where-clause
        prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, \
        p_where, p_wn_best, p_wc_best, p_wo_best, p_wvi_best \
            = self.EG_decoding_where(wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                                     nlu_t, nlu_tt, tt_to_t_idx, nlu,
                                     pr_sc_best, pr_sa_best,
                                     beam_size=4)

        p_tot = cal_prob_tot(p_select, p_where)
        return pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_wvi_best, \
               pr_sql_i, p_tot, p_select, p_where, p_sc_best, p_sa_best, \
               p_wn_best, p_wc_best, p_wo_best, p_wvi_best

    def EG_decoding_select(self, wemb_h, l_hs, tb,
                           beam_size=4, show_p_sc=False, show_p_sa=False):

        # sc
        s_sc = self.scp(wemb_h, l_hs)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        score_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)

        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa
            score_sc_sa[:, i_beam, :] = s_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc]  # [B]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.
        idxs = remap_sc_idx(idxs, pr_sc_beam)  # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:  # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break

        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # output for later analysis.
        p_sc_best = cal_prob_sc(s_sc, pr_sc_best)
        p_sa_best = cal_prob_sa(score_sc_sa[range(bS), beam_idx_sca, :].squeeze(1), pr_sa_best)
        p_select = cal_prob_select(p_sc_best, p_sa_best)
        # p_select  = prob_sca[range(bS),beam_idx_sca,pr_sa_best].detach().to('cpu').numpy()

        return prob_sca, pr_sc_best, pr_sa_best, p_sc_best, p_sa_best, p_select

    def EG_decoding_where(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                          nlu_t, nlu_wp_t, tt_to_t_idx, nlu,
                          pr_sc_best, pr_sa_best,
                          beam_size=4, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        bS, max_header_len, _ = wemb_h.shape

        # Now, Where-clause beam search.
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops

        s_wn = self.wnp(cls_vec)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        idx_st = idx_ed + 1
        idx_ed = idx_st + 1

        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)
        prob_wc = torch.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.n_where_num] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)  # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.n_where_num])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]

        # get most probable n_where_num where-clouses
        # wo
        idx_st = idx_ed + 1
        idx_ed = idx_st + self.n_cond_ops
        s_wo_max = self.wop(wemb_h, pr_wc_max, idx_st, idx_ed)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, n_where_num, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        prob_wvi_beam_st_op_list = []
        prob_wvi_beam_ed_op_list = []

        # To re-use code, repeat the calculation unnecessarily.
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.n_where_num] * bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, pr_wc_max)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam, prob_wvi_beam_st, prob_wvi_beam_ed = pred_wvi_se_beam(self.n_where_num, s_wv,
                                                                                              beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)

            prob_wvi_beam_op_list.append(prob_wvi_beam)
            prob_wvi_beam_st_op_list.append(prob_wvi_beam_st)
            prob_wvi_beam_ed_op_list.append(prob_wvi_beam_ed)
            # pr_wvi_beam = [B, n_where_num, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, n_where_num, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wc_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wo_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_st_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_ed_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])

        for b in range(bS):
            for i_wn in range(self.n_where_num):
                for i_op in range(self.n_cond_ops - 1):  # do not use final one
                    p_wc = prob_wc_max[b, i_wn]
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv
                        prob_wc_dupl[b, i_wn, i_op, i_wv_beam] = p_wc
                        prob_wo_dupl[b, i_wn, i_op, i_wv_beam] = p_wo

                        p_wv_st = prob_wvi_beam_st_op_list[i_op][b, i_wn, i_wv_beam]
                        p_wv_ed = prob_wvi_beam_ed_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_wvi_st_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_st
                        prob_wvi_ed_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_ed

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.n_where_num:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1. Collect only executable one. It is descending order of the probability.
        pr_wvi_max = []

        p_wc_max = []
        p_wo_max = []
        p_wvi_max = []
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            pr_wvi1_max = []

            p_wc1_max = []
            p_wo1_max = []
            p_wvi1_max = []

            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # idx11[0]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [tt_to_t_idx[b]],
                                                             [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wc11_max = prob_wc_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wo11_max = prob_wo_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wvi11_max = [prob_wvi_st_dupl[b, idxs11[0], idxs11[1], idxs11[2]],
                               prob_wvi_ed_dupl[b, idxs11[0], idxs11[1], idxs11[2]]]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc_best[b], pr_sa_best[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
                    pr_wvi1_max.append(wvi)

                    p_wc1_max.append(p_wc11_max)
                    p_wo1_max.append(p_wo11_max)
                    p_wvi1_max.append(p_wvi11_max)

            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)
            pr_wvi_max.append(pr_wvi1_max)

            p_wc_max.append(p_wc1_max)
            p_wo_max.append(p_wo1_max)
            p_wvi_max.append(p_wvi1_max)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []  # total where-clause probability
        pr_wn_based_on_prob = []
        pr_wvi_best = []

        p_wc = []
        p_wo = []
        p_wvi = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_wvi_best1 = pr_wvi_max[b][:pr_wn_based_on_prob[b]]

            pr_sql_i.append(pr_sql_i1)
            pr_wvi_best.append(pr_wvi_best1)

            p_wc.append(p_wc_max[b][:pr_wn_based_on_prob[b]])
            p_wo.append(p_wo_max[b][:pr_wn_based_on_prob[b]])
            p_wvi.append(p_wvi_max[b][:pr_wn_based_on_prob[b]])

        # s_wv = [B, n_where_num, max_nlu_tokens, 2]

        p_wn = cal_prob_wn(s_wn, pr_wn_based_on_prob)
        p_where = cal_prob_where(p_wn, p_wc, p_wo, p_wvi)

        return prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, \
               p_where, p_wn, p_wc, p_wo, p_wvi


def Loss_s2s(score, g_pnt_idxs):
    """
    score = [B, T, max_seq_length]
    """
    #         WHERE string part
    loss = 0

    for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
        ed = len(g_pnt_idxs1) - 1
        score_part = score[b, :ed]
        loss += F.cross_entropy(score_part, torch.tensor(g_pnt_idxs1[1:]).to(device))  # +1 shift.
    return loss


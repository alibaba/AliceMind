# -*- encoding:utf-8 -*-
import os, sys, argparse, re, json
import time
import torch
import random as python_random
from uer.utils.tokenizer import *
from uer.utils.vocab import Vocab
from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from tableModel import TableTextPretraining

import comp_sql
import pandas as pd
# torch.cuda.set_device(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_hyperparam(args):
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)
    args.emb_size = param.get("emb_size", 768)
    args.hidden_size = param.get("hidden_size", 768)
    args.kernel_size = param.get("kernel_size", 3)
    args.block_size = param.get("block_size", 2)
    args.feedforward_size = param.get("feedforward_size", 3072)
    args.heads_num = param.get("heads_num", 12)
    args.layers_num = param.get("layers_num", 12)
    args.dropout = param.get("dropout", 0.1)

    return args

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=3, type=int)
    parser.add_argument("--bS", default=3, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=True,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--task", default='finance_benchmark', type=str,
                        help="Type of model.")
    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='models/google_zh_vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=350, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=1, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--lr_amr', default=1e-4, type=float, help='BERT model learning rate.')
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")

    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")

    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", help="Path of the config file.")
    parser.add_argument("--vocab_path", type=str, default="./models/google_zh_vocab.txt",
                        help="Path of the vocabulary file.")
    parser.add_argument("--mlp_arc_size", type=int, default=400, help="batch size.")
    parser.add_argument("--mlp_rel_size", type=int, default=100, help="batch size.")

    parser.add_argument("--table_bert_dir", default='/Users/yuchen/Downloads/_models/ptm/0830_ptm_base.bin-20000', type=str, help="table_bert")
    parser.add_argument("--data_dir", default='./data/cbank', type=str, help="table_bert")
    parser.add_argument("--train_name", default='0901_train_cbank.txt',
                        type=str, help="table_bert")
    parser.add_argument("--dev_name", default='0901_dev_cbank.txt',
                        type=str, help="table_bert")
    parser.add_argument("--test_name", default='0901_test_cbank.txt',
                        type=str, help="table_bert")
    parser.add_argument("--table_name", default='cbank_table.json',
                        type=str, help="table_bert")
    parser.add_argument("--table_words", default='cbank_value_name.csv',
                        type=str, help="table_bert")

    # 1.5 auto train args
    parser.add_argument("--bert_path", default='./model/ERNIE', type=str,
                          help='config path to use (e.g. ./conf/config)')
    parser.add_argument("--filename", default='./example/train.zip', type=str)
    parser.add_argument("--output_dir", default='model/tableqa/', type=str)
    parser.add_argument("--job_id", default='nl2sql001', type=str)
    parser.add_argument("--heartbeat_host", default='127.0.0.1', type=str)
    parser.add_argument("--heartbeat_port", default=8880, type=int)

    args = parser.parse_args()

    args.target = "bert"

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    # print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("make new folder ", path)

def get_table_bert(args):
    bert_config = args
    args.num_hidden_layers = args.layers_num

    args.pretrained_model_path = args.table_bert_dir
    uer_tokenizer = BertTokenizer(args)

    table_bert_model = TableTextPretraining(args)

    if args.use_cuda:
        table_bert_model_dict = torch.load(args.pretrained_model_path)
    else:
        table_bert_model_dict = torch.load(args.pretrained_model_path, map_location='cpu')
    table_bert_model_dict = {k: v for k, v in table_bert_model_dict.items() if k in table_bert_model.state_dict()}
    table_bert_model.load_state_dict(table_bert_model_dict, strict=False)
    # print('model bert:', table_bert_model)
    print("Load pre-trained parameters.")
    return table_bert_model.pre_encoder, uer_tokenizer, bert_config


def get_opt(args, model, model_bert, fine_tune, total_steps):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = None

    return opt, opt_bert


def get_models(args, trained=False):
    # some constants
    #agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    #cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
    dep_ops = ['null', 'scol', 'agg', 'wcol', 'val', 'op', 'sort_col', 'sort_op', 'sort_value']
    agg_ops = ["", "AVG", "MAX", "MIN", "COUNT", "SUM", "COMPARE", "GROUP BY", "SAME"]
    cond_ops = [">", "<", "==", "!=", "ASC", "DESC"]

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    table_bert, tokenizer, bert_config = get_table_bert(args)

    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL
    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None
        print(".......")
        print("loading from ", path_model_bert, " and ", path_model, " and ", path_model_amr)
        print(".......")
        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        table_bert.load_state_dict(res['model_bert'])
        table_bert.to(device)
        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, table_bert, tokenizer, bert_config

def loadTableWords(args):
    value_name_path = os.path.join(args.data_dir, args.table_words)
    if os.path.exists(value_name_path):
        star_words = pd.read_table(value_name_path, header=0)
        # print(star_words.head())
        rowx, y = star_words.shape
        for idx in range(rowx):
            star_words.loc[idx]['归一化列值'] = str(star_words.loc[idx]['归一化列值']).replace(' ', '')
    else:
        star_words = None
    return star_words

def load_nl2sql_bussiness(path_nl2sql, mode, bussiness_name):
    """ Load training sets
        """
    sub_dir = mode  # here!!!!!!!
    path_dir = path_nl2sql + '/' + bussiness_name + '/'
    path_data = path_dir + mode + '_' + bussiness_name+'_tok.json'
    path_data = path_dir + 'uer_' +mode + '_' + bussiness_name+'_tok.json'

    path_table = path_dir + bussiness_name + '_table.json'
    #path_table = path_nl2sql + 'alltables.json'
    print("path_data: ", path_data)
    print("path_table:", path_table)

    data = []
    table = {}
    with open(path_data, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            #t1 = json.loads(line)
            t1 = eval(line)
            data.append(t1)

    with open(path_table, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            #t1 = eval(line.strip())
            table[t1['tablename']] = t1

    return data, table


def get_bussiness_data(path_nl2sql, args):
    train_data, train_table = load_nl2sql_bussiness(path_nl2sql, 'train', bussiness_name=args.task)
    val_data, val_table = load_nl2sql_bussiness(path_nl2sql, 'test', bussiness_name=args.task)
    train_loader, dev_loader = get_loader_wikisql(train_data, val_data, args.bS, shuffle_train=True)

    return train_data, train_table, val_data, val_table, train_loader, dev_loader


def get_yewu_single_data(args):
    train_data = []
    val_data = []
    test_data = []
    tables = {}
    shuffle_train = True
    with open(os.path.join(args.data_dir, args.train_name), encoding='utf-8') as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line)
            train_data.append(t1)

    with open(os.path.join(args.data_dir, args.dev_name), encoding='utf-8') as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line)
            val_data.append(t1)

    with open(os.path.join(args.data_dir, args.test_name), encoding='utf-8') as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line)
            test_data.append(t1)

    with open(os.path.join(args.data_dir, args.table_name), encoding='utf-8') as f:
        for line in f.readlines():
            table = eval(line.strip())
            # table = json.dumps(line.strip())
            if 'tablename' in table:
                tid = table['tablename']
            else:
                tid = table['id']
                table['tablename'] = table['id']
            if 'types' in table:
                table['col_types'] = table['types']
                table['col_types'] = [x.lower() for x in table['col_types']]
                table['types'] = [x.lower() for x in table['types']]
            else:
                table['col_types'] = [x.lower() for x in table['col_types']]
                table['types'] = [x.lower() for x in table['col_types']]

            if 'header' in table:
                table['headers'] = table['header']
            if 'headers' in table:
                table['header'] = table['headers']
            if 'unit' in table:
                pass
            else:
                table['unit'] = ['Null'] * len(table['headers'])


            tables[tid] = table


    print('lengths:', len(train_data), len(val_data), len(test_data))


    train_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=train_data,
        shuffle=shuffle_train,
        num_workers=1,
        collate_fn=temp_func  # now dictionary values are not merged!
    )
    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=val_data,
        shuffle=shuffle_train,
        num_workers=1,
        collate_fn=temp_func  # now dictionary values are not merged!
    )
    test_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=test_data,
        shuffle=shuffle_train,
        num_workers=1,
        collate_fn=temp_func  # now dictionary values are not merged!
    )


    return train_data, val_data, test_data, tables, train_loader, dev_loader, test_loader



def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer, epoch,
           task, max_seq_length, num_target_layers, accumulate_gradients=1, start_time=None, heartbeat_hook=None, 
           callconfig=None, check_grad=True, st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    if dset_name == 'train':
        model.train()
        model_bert.train()
    else:
        model.eval()
        model_bert.eval()

    amr_loss = 0
    ave_loss = 0
    slen_loss = 0
    sc_loss = 0
    scco_loss = 0
    sa_loss = 0
    wn_loss = 0
    wc_loss = 0
    wo_loss = 0
    wvi_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_scco = 0
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_lx_r = 0
    cnt_x = 0  # of execution acc
    right_sql_cnt = 0
    sql_acc = 0.0


    # Engine for SQL querying.
    # engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    #engine = DBEngine(path_db)
    # print(train_table[0])
    if dset_name == 'train':
        epoch_start_time = time.time()
        for iB, t in enumerate(train_loader):
            torch.cuda.empty_cache()
            if iB % 100 == 0:
                print(iB, "/", len(train_loader), "\tUsed time:", time.time() - epoch_start_time, "\tloss:", ave_loss/(iB+0.00001),)
            sys.stdout.flush()
            cnt += len(t)
            if cnt < st_pos:
                continue
            # Get fields
            nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_cond_conn_op, g_slen, idxs = get_g(sql_i)
            # g_wvi = get_g_wvi_corenlp(t, idxs)
            wemb_n, wemb_h, l_n, l_hpu, l_hs = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
            g_wvi_corenlp = get_g_wvi_corenlp(t, idxs)
            try:
                g_wvi = g_wvi_corenlp
            except:
                print('????wvi')
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                # e.g. train: 32.
                continue
            knowledge = []
            for k in t:
                if "bertindex_knowledge" in k:
                    knowledge.append(k["bertindex_knowledge"])
                else:
                    knowledge.append(max(l_n)*[0])
            knowledge_header = []
            for k in t:
                if "header_knowledge" in k:
                    knowledge_header.append(k["header_knowledge"])
                else:
                    knowledge_header.append(max(l_hs) * [0])
            # score
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_cco, s_slen = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                                      g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc,
                                                                      g_wvi=g_wvi,
                                                                      g_cond_conn_op=g_cond_conn_op, g_slen=g_slen,
                                                                      knowledge = knowledge, knowledge_header = knowledge_header)
            loss, loss_slen, loss_sc, loss_scco, loss_sa, loss_wn, loss_wc, loss_wo, loss_wvi = Loss_sw_se(s_sc, s_cco,
                                                                                                           s_sa, s_wn,
                                                                                                           s_wc, s_wo,
                                                                                                           s_wv, s_slen,
                                                                                                           g_sc, g_sa,
                                                                                                           g_wn, g_wc,
                                                                                                           g_wo, g_wvi,
                                                                                                           g_cond_conn_op,
                                                                                                           g_slen)
            loss_all = loss
            if dset_name == 'dev':
                pass
            else:
                # Calculate gradient
                if iB % accumulate_gradients == 0:  # mode
                    # at start, perform zero_grad
                    opt.zero_grad()
                    if opt_bert:
                        opt_bert.zero_grad()
                    loss_all.backward()
                    if accumulate_gradients == 1:
                        opt.step()
                        if opt_bert:
                            opt_bert.step()
                elif iB % accumulate_gradients == (accumulate_gradients - 1):
                    # at the final, take step with accumulated graident
                    loss_all.backward()
                    opt.step()
                    if opt_bert:
                        opt_bert.step()
                else:
                    # at intermediate stage, just accumulates the gradients
                    loss_all.backward()
            # Prediction
            pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_slen = pred_sw_se(s_sc, s_cco, s_sa, s_wn, s_wc,
                                                                                     s_wo, s_wv, s_slen)

            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu)
            # print("agg:", pr_sa, ",cond_conn_op:", pr_scco, ",sel:", pr_sc, ",conds:" ,pr_wc, pr_wo, pr_wv_str)
            # print("@@@@@@@@@@@@@@")

            pr_sql_i = generate_sql_i(pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu, t, train_table)

            for k in range(len(sql_i)):
                try:
                    if (np.sort(np.asarray(sql_i[k]['conds']), axis=0) == np.sort(np.asarray(pr_sql_i[k]['conds']),
                                                                                  axis=0)).all() and \
                            (sql_i[k]['sel'] == np.asarray(pr_sql_i[k]['sel'])).all() and \
                            (sql_i[k]['agg'] == np.asarray(pr_sql_i[k]['agg'])).all() and \
                            (sql_i[k]['cond_conn_op'] == pr_sql_i[k]['cond_conn_op']):
                        cnt_lx_r += 1
                    else:
                        pass
                except:
                    cnt_lx_r += 1
                if (pr_wc[k] == g_wc[k]) is False:
                    pass

            # Cacluate accuracy
            cnt_sc1_list, cnt_scco1_list, cnt_sa1_list, cnt_wn1_list, \
            cnt_wc1_list, cnt_wo1_list, \
            cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_cond_conn_op, g_sa, g_wn, g_wc, g_wo, g_wvi, g_slen,
                                                          pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                          sql_i, pr_sql_i,
                                                          mode='train')

            cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_scco1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                           cnt_wo1_list, cnt_wv1_list)
            # statistics
            ave_loss += loss.item()
            slen_loss += loss_slen.item()
            sc_loss += loss_sc.item()
            sa_loss += loss_sa.item()
            scco_loss += loss_scco.item()
            wn_loss += loss_wn.item()
            wc_loss += loss_wc.item()
            wo_loss += loss_wo.item()
            wvi_loss += loss_wvi.item()

            # count
            cnt_sc += sum(cnt_sc1_list)
            cnt_scco += sum(cnt_scco1_list)
            cnt_sa += sum(cnt_sa1_list)
            cnt_wn += sum(cnt_wn1_list)
            cnt_wc += sum(cnt_wc1_list)
            cnt_wo += sum(cnt_wo1_list)
            cnt_wvi += sum(cnt_wvi1_list)
            cnt_wv += sum(cnt_wv1_list)
            cnt_lx += sum(cnt_lx1_list)
            # cnt_x += sum(cnt_x1_list)
    else:
        with torch.no_grad():
            table_words = loadTableWords(args)
            for iB, t in enumerate(train_loader):
                torch.cuda.empty_cache()


                cnt += len(t)

                if cnt < st_pos:
                    continue

                # nlu, nlu_t, sql_i, tb, hs_t = get_fields_info(t, train_table)
                # nlu  : natural language utterance
                # nlu_t: tokenized nlu
                # sql_i: canonical form of SQL query
                # sql_q: full SQL query text. Not used.
                # sql_t: tokenized SQL query
                # tb   : table
                # hs_t : tokenized headers. Not used.

                nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)

                g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_cond_conn_op, g_slen, idxs = get_g(sql_i)

                # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
                # g_wvi = get_g_wvi_corenlp(t, idxs)

                # wemb_n, wemb_h, l_n, l_hpu, l_hs = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hs_t, max_seq_length,
                #                     num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

                wemb_n, wemb_h, l_n, l_hpu, l_hs = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds,
                                                                 max_seq_length,
                                                                 num_out_layers_n=num_target_layers,
                                                                 num_out_layers_h=num_target_layers)

                g_wvi_corenlp = get_g_wvi_corenlp(t, idxs)
                try:
                    g_wvi = g_wvi_corenlp
                except:
                    # Exception happens when where-condition is not found in nlu_tt.
                    # In this case, that train example is not used.
                    # During test, that example considered as wrongly answered.
                    # e.g. train: 32.
                    print('wvi???')
                    continue
                knowledge = []
                for k in t:
                    if "bertindex_knowledge" in k:
                        knowledge.append(k["bertindex_knowledge"])
                    else:
                        knowledge.append(max(l_n)*[0])

                knowledge_header = []
                for k in t:
                    if "header_knowledge" in k:
                        knowledge_header.append(k["header_knowledge"])
                    else:
                        knowledge_header.append(max(l_hs) * [0])

                # score
                s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_cco, s_slen = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                                          g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc,
                                                                          g_wvi=g_wvi,
                                                                          g_cond_conn_op=g_cond_conn_op, g_slen=g_slen,
                                                                          knowledge = knowledge, knowledge_header = knowledge_header)

                loss, loss_slen, loss_sc, loss_scco, loss_sa, loss_wn, loss_wc, loss_wo, loss_wvi = Loss_sw_se(s_sc,
                                                                                                               s_cco,
                                                                                                               s_sa,
                                                                                                               s_wn,
                                                                                                               s_wc,
                                                                                                               s_wo,
                                                                                                               s_wv,
                                                                                                               s_slen,
                                                                                                               g_sc,
                                                                                                               g_sa,
                                                                                                               g_wn,
                                                                                                               g_wc,
                                                                                                               g_wo,
                                                                                                               g_wvi,
                                                                                                               g_cond_conn_op,
                                                                                                               g_slen)

                # Prediction
                pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_slen = pred_sw_se(s_sc, s_cco, s_sa, s_wn, s_wc,
                                                                                         s_wo, s_wv, s_slen)

                pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu)
                # print("agg:", pr_sa, ",cond_conn_op:", pr_scco, ",sel:", pr_sc, ",conds:" ,pr_wc, pr_wo, pr_wv_str)
                # print("@@@@@@@@@@@@@@")

                pr_sql_i = generate_sql_i(pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu, t, train_table)
                for k in range(len(sql_i)):
                    cond_com_flag = comp_sql.com_conds(sql_i[k], pr_sql_i[k], tb[k], table_words)
                    sel_com_flag = comp_sql.com_sels_with_split(g_sc[k], g_sa[k], pr_sql_i[k], tb[k], table_words)
                    if cond_com_flag and sel_com_flag:
                        right_sql_cnt += 1
                    try:
                        if (np.sort(np.asarray(sql_i[k]['conds']), axis=0) == np.sort(
                                np.asarray(pr_sql_i[k]['conds']), axis=0)).all() and \
                                (sql_i[k]['sel'] == np.asarray(pr_sql_i[k]['sel'])).all() and \
                                (sql_i[k]['agg'] == np.asarray(pr_sql_i[k]['agg'])).all() and \
                                (sql_i[k]['cond_conn_op'] == pr_sql_i[k]['cond_conn_op']):
                            cnt_lx_r += 1
                        else:
                            pass
                    except:
                        cnt_lx_r += 1
                    if pr_wc[k] == g_wc[k] is False:
                        pass

                # Cacluate accuracy
                cnt_sc1_list, cnt_scco1_list, cnt_sa1_list, cnt_wn1_list, \
                cnt_wc1_list, cnt_wo1_list, \
                cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_cond_conn_op, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                              g_slen,
                                                              pr_sc, pr_scco, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                              sql_i, pr_sql_i,
                                                              mode='train')

                cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_scco1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                               cnt_wo1_list, cnt_wv1_list)

                # statistics
                ave_loss += loss.item()
                slen_loss += loss_slen.item()
                sc_loss += loss_sc.item()
                sa_loss += loss_sa.item()
                scco_loss += loss_scco.item()
                wn_loss += loss_wn.item()
                wc_loss += loss_wc.item()
                wo_loss += loss_wo.item()
                wvi_loss += loss_wvi.item()

                # count
                cnt_sc += sum(cnt_sc1_list)
                cnt_scco += sum(cnt_scco1_list)
                cnt_sa += sum(cnt_sa1_list)
                cnt_wn += sum(cnt_wn1_list)
                cnt_wc += sum(cnt_wc1_list)
                cnt_wo += sum(cnt_wo1_list)
                cnt_wvi += sum(cnt_wvi1_list)
                cnt_wv += sum(cnt_wv1_list)
                cnt_lx += sum(cnt_lx1_list)
                # cnt_x += sum(cnt_x1_list)
                # break
            sql_acc = right_sql_cnt/cnt
            print('sql_acc:', right_sql_cnt/cnt)

    amr_loss /= cnt
    ave_loss /= cnt
    slen_loss /= cnt
    sc_loss /= cnt
    sa_loss /= cnt
    scco_loss /= cnt
    wn_loss /= cnt
    wc_loss /= cnt
    wo_loss /= cnt
    wvi_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_scco = cnt_scco / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_lx_r = cnt_lx_r / cnt
    print(
        'Epoch {}, slen_loss = {}, sc_loss = {}, sa_loss = {}, scco_loss = {}, wn_loss = {}, wc_loss = {}, wo_loss = {}, wvi_loss = {}'.format(
            epoch, slen_loss, sc_loss, sa_loss, scco_loss, wn_loss, wc_loss, wo_loss, wvi_loss))
    # print('cnt_lx_r = {}'.format(cnt_lx_r))
    # acc_x = cnt_x / cnt

    # acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    acc = [ave_loss, acc_sc, acc_scco, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx]

    aux_out = 1

    return acc, aux_out, sql_acc

def save_model(args, model, model_amr, model_bert):
    state = {'model': model.state_dict()}
    torch.save(state, os.path.join(args.output_dir, 'model_best.pt'))

    state = {'model_amr': model_amr.model.state_dict()}
    torch.save(state, os.path.join(args.output_dir, 'model_amr_best.pt'))

    state = {'model_bert': model_bert.state_dict()}
    torch.save(state, os.path.join(args.output_dir, 'model_bert_best.pt'))

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_scco, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_scco: {acc_scco:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}"
    )

if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    # args.vocab_size = 21128
    args = load_hyperparam(args)
    pre_vocab = Vocab()
    pre_vocab.load(args.vocab_path)
    args.vocab = pre_vocab
    args.vocab_size = len(pre_vocab)

    ## 2. Paths
    BERT_PT_PATH = './model/ERNIE'
    ## 3. Load data
    # train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_bussiness_data(path_nl2sql, args)
    train_data, val_data, test_data, tables, train_loader, dev_loader, test_loader = get_yewu_single_data(args)

    ## 4. Build & Load models
    # To start from the pre-trained models, un-comment following lines.
    num_train_optimization_steps = int(len(train_data) / args.bS / args.accumulate_gradients) * args.tepoch
    model, model_bert, tokenizer, bert_config = get_models(args, trained=False)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(args, model, model_bert, args.fine_tune, num_train_optimization_steps)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    sql_acc_best = 0.0
    for epoch in range(args.tepoch):
        # train
        print("")
        print("*************")
        print("*************")
        if epoch>0:
            for p in opt.param_groups:
                p['lr'] *= 0.9
            for p in opt_bert.param_groups:
                p['lr'] *= 0.9
        acc_train, aux_out_train, sql_acc_train = train(train_loader, tables, model, model_bert, opt, bert_config, tokenizer, epoch,
                                          args.task, args.max_seq_length, args.num_target_layers, args.accumulate_gradients, start_time=None,
                                           heartbeat_hook=None, callconfig=None, opt_bert=opt_bert, st_pos=0, path_db=None, dset_name='train')

        # check DEV
        print("")
        print("@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@")

        acc_dev, aux_out_dev, sql_acc_dev = train(dev_loader, tables, model, model_bert, opt, bert_config, tokenizer, epoch,
                                      args.task, args.max_seq_length, args.num_target_layers, args.accumulate_gradients, start_time=None,
                                       heartbeat_hook=None, callconfig=None, opt_bert=opt_bert, st_pos=0, path_db=None, dset_name='dev')
        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')
        acc_lx_t = acc_dev[-1]
        if sql_acc_dev > sql_acc_best:
            sql_acc_best = sql_acc_dev
            epoch_best = epoch
            print(f" Best Dev sql acc: {sql_acc_best} at epoch: {epoch_best}")
            acc_test, aux_out_test, sql_acc_test = train(test_loader, tables, model, model_bert, opt, bert_config, tokenizer, epoch,
                                      args.task, args.max_seq_length, args.num_target_layers, args.accumulate_gradients, start_time=None,
                                       heartbeat_hook=None, callconfig=None, opt_bert=opt_bert, st_pos=0, path_db=None, dset_name='dev')
            print(f" Best Test sql acc: {sql_acc_test} at epoch: {epoch_best}")





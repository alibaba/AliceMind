# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 10:41 AM
# @Author  : yuchen
# @FileName: comp_sql.py
# @Software: PyCharm

import numpy as np
import os
import pandas as pd
import pickle
import copy
## syn, syn_product, unit, konglie
from collections import Counter

class Syn():
    def __init__(self, args):
        self.all_syns_dict, self.col_scripts_dict, self.jianyu_syns_dict, self.m2e_syns_dict, self.crowd_syns_dict, self.crowd_scripts_dict, self.crowd_shifou_dict = self.load_syns_scripts()

        table_words = self.loadTableWords()

        self.args = args



    def load_syns_scripts(self):
        zh_dir = './zh_kb_syn_dict'
        with open(os.path.join(zh_dir, 'all_syns.pickle'), 'rb') as f:
            all_syns_dict = pickle.load(f)
        with open(os.path.join(zh_dir, 'm2e_syns0125.pickle'), 'rb') as f:
            m2e_syns_dict = {}

        with open(os.path.join(zh_dir, 'col_scripts.pickle'), 'rb') as f:
            col_scripts_dict = pickle.load(f)

        with open(os.path.join(zh_dir, 'jianyu_syns.pickle'), 'rb') as f:
            jianyu_syns_dict = pickle.load(f)

        with open(os.path.join(zh_dir, 'syn_from_crowd0526.pickle'), 'rb') as f:
            crowd_syns_dict = pickle.load(f)

        with open(os.path.join(zh_dir, 'scripts_from_crowd0524.pickle'), 'rb') as f:
            crowd_scripts_dict = pickle.load(f)

        with open(os.path.join(zh_dir, 'shifou_from_crowd0524.pickle'), 'rb') as f:
            crowd_shifou_dict = pickle.load(f)

        # print('scropts:')
        # for k, v in col_scripts_dict.items():
        #     print(k, v)

        return all_syns_dict, col_scripts_dict, jianyu_syns_dict, m2e_syns_dict, crowd_syns_dict, crowd_scripts_dict, crowd_shifou_dict


    def loadTableWords(self):
        value_name_path = '/Users/yuchen/Documents/Workroom1/product_taccs/data/car/benz/benz_value_name.csv'
        value_name_path = os.path.join(self.args.data_dir, self.args.table_words)
        if os.path.exists(value_name_path):
            star_words = pd.read_table(value_name_path, header=0)
            # print(star_words.head())
            rowx, y = star_words.shape
            for idx in range(rowx):
                star_words.loc[idx]['归一化列值'] = str(star_words.loc[idx]['归一化列值']).replace(' ', '')
        else:
            star_words = None
        return star_words



def col_val_syn(table, table_words, col_index, val):
    # table = self.tables[tableId]
    val = str(val)

    headers = table['header']

    cond_col = headers[col_index]

    cond_value_synonmys = []
    cond_value_synonmys.append(val)
    cond_value_synonmys.append(val.lower())
    cond_value_synonmys.append(val+'的')


    if table_words is None:
        return cond_value_synonmys
    ## 单位
    sel_unit = table['unit'][col_index]
    if sel_unit != 'Null' and sel_unit != "":
        sel_units = str(sel_unit).split('|')
        # sel_uniti = random.choice(sel_units)
        for sel_uniti in sel_units:
            value_unit = str(val) + sel_uniti
            cond_value_synonmys.append(value_unit)

    for index, row in table_words.iterrows():
        if row['列名'] == cond_col and str(row['归一化列值']) == val and (pd.isnull(row['同义词']) == False):
            cond_value_synonmys += row['同义词'].split('|')

    return cond_value_synonmys



def com_conds(gold_sql, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_conds = gold_sql['conds'].tolist()
    pre_conds1 = copy.deepcopy(pr_sql['conds'])
    # print('gold conds:', gold_conds)

    pre_conds = []
    for x in pre_conds1:
        pre_idx, pre_op, pre_val = x[:3]
        pre_val = str(pre_val)
        if pre_idx < len(tablei['header']) and tablei['types'][pre_idx] in ['text', 'bool']:
            if pre_op not in [2, 3]:
                x[1] = 2
        if x not in pre_conds:
            pre_conds.append(x)

    # print('pre conds:', pre_conds)


    if len(gold_conds) != len(pre_conds):
        return False

    ## step 1: null cond:
    if len(gold_conds) == 1 and int(gold_conds[0][0]) == len(tablei['header']):
        # for pre_condi in pre_conds:
        #     if pre_condi[0] == len(tablei['header']):
        #         return True
        if len(pre_conds) == 1 and pre_conds[0][0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        ## step2 : 没有空条件，考虑同义词和单位
        for gold_condsi in gold_conds:
            # print('condi:', gold_condsi)
            cond_idx, cond_op, val, val_syn = gold_condsi[:4]
            cond_idx, cond_op = int(cond_idx), int(cond_op)
            val = str(val)
            val_syn = str(val_syn)
            table_syns = col_val_syn(tablei, table_words, int(gold_condsi[0]), gold_condsi[2])
            # print('table syns：', table_syns)
            # print('kb syns:', kb_syns)
            all_cond_syns = table_syns + [val_syn]
            all_cond_syns_low = [x.lower() for x in all_cond_syns]
            all_cond_syns += all_cond_syns_low

            find_flag = False
            for pre_condi in pre_conds:
                pre_idx, pre_op, pre_val = pre_condi[:3]
                pre_val = str(pre_val)
                pre_idx, pre_op = int(pre_idx), int(pre_op)
                val_right = False
                for val_syni in all_cond_syns:
                    if val_syni in pre_val or pre_val in val_syni or pre_val == val_syni:
                        val_right = True

                if pre_idx == cond_idx and pre_op == cond_op and (pre_val in all_cond_syns or val in pre_val or val_right):
                    find_flag = True
            if not find_flag:
                return False
        return True

def com_sels(gold_sql, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_sels = gold_sql['sel'].tolist()
    gold_aggs = gold_sql['agg'].tolist()
    pre_sels = pr_sql['sel']
    pre_aggs = pr_sql['agg']

    if len(gold_sels) != len(pre_sels):
        return False

    if len(gold_sels) == 1 and gold_sels[0] == len(tablei['header']):
        if len(pre_sels) == 1 and pre_sels[0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        gold_sel_aggs = zip(gold_sels, gold_aggs)
        pre_sel_aggs = zip(pre_sels, pre_aggs)
        for idx, gold_sel_aggi in enumerate(gold_sel_aggs):
            if gold_sel_aggi in pre_sel_aggs:
                pass
            else:
                return False
        return True


def com_conds_final(gold_sql, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_conds = gold_sql['conds']
    pre_conds1 = copy.deepcopy(pr_sql['conds'])
    # print('gold conds:', gold_conds)

    pre_conds = []
    for x in pre_conds1:
        pre_idx, pre_op, pre_val = x[:3]
        if pre_idx < len(tablei['header']) and tablei['types'][pre_idx] in ['text', 'bool']:
            if pre_op not in [2, 3]:
                x[1] = 2
        if x not in pre_conds:
            pre_conds.append(x)

    # print('pre conds:', pre_conds)


    if len(gold_conds) != len(pre_conds):
        return False

    ## step 1: null cond:
    if len(gold_conds) == 1 and int(gold_conds[0][0]) == len(tablei['header']):
        # for pre_condi in pre_conds:
        #     if pre_condi[0] == len(tablei['header']):
        #         return True
        if len(pre_conds) == 1 and pre_conds[0][0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        ## step2 : 没有空条件，考虑同义词和单位
        for gold_condsi in gold_conds:
            # print('condi:', gold_condsi)
            cond_idx, cond_op, val, val_syn = gold_condsi[:4]
            cond_idx = int(cond_idx)
            if cond_idx < len(tablei['header']):
                val = str(val)
                val_syn = str(val_syn)
                cond_idx, cond_op = int(cond_idx), int(cond_op)
                table_syns = col_val_syn(tablei, table_words, int(gold_condsi[0]), gold_condsi[2])
                # print('table syns：', table_syns)
                # print('kb syns:', kb_syns)
                all_cond_syns = table_syns + [val_syn]
                all_cond_syns_low = [x.lower() for x in all_cond_syns]
                all_cond_syns += all_cond_syns_low

                find_flag = False
                for pre_condi in pre_conds:
                    pre_idx, pre_op, pre_val = pre_condi[:3]
                    pre_val = str(pre_val)
                    pre_idx, pre_op = int(pre_idx), int(pre_op)
                    val_right = False
                    for val_syni in all_cond_syns:
                        if val_syni in pre_val or pre_val in val_syni or pre_val == val_syni:
                            val_right = True

                    if pre_idx == cond_idx and pre_op == cond_op and (pre_val in all_cond_syns or val in pre_val or val_right):
                        find_flag = True
                if not find_flag:
                    return False
        return True

def com_sels_final(gold_sql, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_sels = gold_sql['sel'].tolist()
    gold_aggs = gold_sql['agg'].tolist()

    pre_sels = pr_sql['sel']
    pre_aggs = pr_sql['agg']

    if len(gold_sels) != len(pre_sels):
        return False

    if len(gold_sels) == 1 and gold_sels[0] == len(tablei['header']):
        if len(pre_sels) == 1 and pre_sels[0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        gold_sel_aggs = zip(gold_sels, gold_aggs)
        pre_sel_aggs = zip(pre_sels, pre_aggs)
        for idx, gold_sel_aggi in enumerate(gold_sel_aggs):
            if gold_sel_aggi in pre_sel_aggs:
                pass
            else:
                return False
        return True



def com_sels_with_split(gold_sql_sc, gold_sql_sa, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_sels = gold_sql_sc
    gold_aggs = gold_sql_sa

    pre_sels = pr_sql['sel']
    pre_aggs = pr_sql['agg']

    if len(gold_sels) != len(pre_sels):
        return False

    if len(gold_sels) == 1 and gold_sels[0] == len(tablei['header']):
        if len(pre_sels) == 1 and pre_sels[0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        gold_sel_aggs = zip(gold_sels, gold_aggs)
        pre_sel_aggs = list(zip(pre_sels, pre_aggs))
        for idx, gold_sel_aggi in enumerate(gold_sel_aggs):
            if gold_sel_aggi in pre_sel_aggs:
                pass
            else:
                return False
        return True


def com_sels_with_split_final(gold_sql_sc, gold_sql_sa, pr_sql, tablei, table_words):
    # print('tablei:', tablei)
    gold_sels = gold_sql_sc
    gold_aggs = gold_sql_sa

    pre_sels = pr_sql['sel']
    pre_aggs = pr_sql['agg']

    if len(gold_sels) != len(pre_sels):
        return False

    if len(gold_sels) == 1 and gold_sels[0] == len(tablei['header']):
        if len(pre_sels) == 1 and pre_sels[0] == len(tablei['header']):
            return True
        else:
            return False
    else:
        gold_sel_aggs = zip(gold_sels, gold_aggs)
        pre_sel_aggs = list(zip(pre_sels, pre_aggs))
        # print(pre_sel_aggs)
        for idx, gold_sel_aggi in enumerate(gold_sel_aggs):
            if gold_sel_aggi in pre_sel_aggs:
                pass
            else:
                # print('not', gold_sel_aggi)
                return False
        return True









if __name__ == '__main__':
    gold = {'agg': [4], 'sel': [0], 'cond_conn_op': 1, 'conds': [['6', '3', '9.3', '9.3'],
                                                                                    ['8', '2', 'False', '不带']],
                                                                                    'use_add_value': 0}
    pre = {'agg': [0, 0, 0, 0], 'cond_conn_op': 2, 'sel': [2, 7, 9, 10],
          'conds': [[0, 5, '焉[SEP]选汽车类型当产品名称等于'], [4, 5, '没']]}

    pre = {'agg': [4], 'sel': [0], 'cond_conn_op': 1, 'conds': [['6', '3', '9.3', '9.3'],
                                                                                    ['8', '2', '没有']],
                                                                                    'use_add_value': 0}

    gold = {'agg': [0, 4, 0], 'sel': [0, 9, 4], 'cond_conn_op': 0, 'conds': [[6, 2, '8.4', '8.4']], 'use_add_value': 0}
    pre = {'agg': [0, 0, 4], 'cond_conn_op': 0, 'sel': [0, 4, 9], 'conds': [[6, 2, '8.4']]}

    tablei = {'tablename': 'benz',
             'header': ['产品名称', '汽车类型', '综合耗油量', '排量', '轴距', '功率', '百公里加速', '全景天窗', '倒车影像', '智能泊车', '零售价'],
             'types': ['text', 'text', 'number', 'number', 'number', 'number', 'number', 'bool', 'bool', 'bool',
                       'number'], 'unit': ['', '', 'L/百公里', '毫升', '毫米', '千瓦|KW', '秒', '', '', '', '万元|万'],
             'attribute': ['PRIMARY', 'KEY', 'KEY', 'MODIFIER', 'MODIFIER', 'MODIFIER', 'KEY', 'MODIFIER', 'MODIFIER',
                           'MODIFIER', 'KEY'],
             'rows': [['奔驰B180时尚型', '轿车', '5.7', '1332', '2729', '136', '9.3', 'False', 'False', 'True', '23.98'],
                      ['奔驰C260', '轿车', '7.3', '1497', '2840', '184', '8.4', 'True', 'False', 'False', '36.68'],
                      ['奔驰E300', '轿车', '7.5', '1991', '2873', '258', '6.5', 'True', 'False', 'False', '59.88'],
                      ['奔驰E53', '轿车', '8.7', '2999', '2873', '435', '4.4', 'True', 'False', 'False', '97.88'],
                      ['奔驰S63', '轿车', '10.0', '3982', '2945', '612', '3.5', 'True', 'False', 'False', '229.88'],
                      ['奔驰GLA180', 'SUV', '6.0', '1332', '2729', '163', '9.9', 'True', 'True', 'True', '27.68'],
                      ['奔驰GLC260动感型', 'SUV', '8.1', '1991', '2973', '197', '8.4', 'True', 'True', 'True', '39.78'],
                      ['奔驰GLC300', 'SUV', '8.75', '1991', '2873', '258', '6.7', 'False', 'False', 'False', '51.88'],
                      ['奔驰GLE53', 'SUV', '9.8', '2999', '2995', '435', '5.5', 'True', 'False', 'False', '102.88'],
                      ['奔驰GLS600礼尚版', 'SUV', '12.0', '3982', '3135', '557', '4.9', 'True', 'False', 'False', '273.8']]}


    is_right = com_conds_final(gold, pre, tablei,None)
    is_right2 = com_sels_with_split_final(gold['sel'], gold['agg'], pre, tablei, None)
    print(is_right)
    print(is_right2)


# Copyright 2019-present NAVER Corp.
# Apache License v2.0


# Wonseok Hwang
# Convert the wikisql format to the suitable format for the BERT.
import os, sys, json
from matplotlib.pylab import *


def get_squad_style_ans(nlu, sql):
    conds = sql['conds']
    answers = []
    for cond1 in conds:
        a1 = {}
        wv1 = cond1[2]
        a1['text'] = wv1
        a1['answer_start'] = nlu.lower().find(str(wv1).lower())
        if a1['answer_start'] < 0 or a1['answer_start'] >= len(nlu):
            raise EnvironmentError
        answers.append(a1)

    return answers


def get_qas(path_q, tid):
    qas = []
    with open(path_q, 'r') as f_q:
        qnum = -1
        for j, q1 in enumerate(f_q):
            q1 = json.loads(q1)
            tid_q = q1['table_id']

            if tid_q != tid:
                continue
            else:
                qnum += 1
                #                 print(tid_q, tid)
                qas1 = {}
                nlu = q1['question']
                sql = q1['sql']

                qas1['question'] = nlu
                qas1['id'] = f'{tid_q}-{qnum}'
                qas1['answers'] = get_squad_style_ans(nlu, sql)
                qas1['c_answers'] = sql

                qas.append(qas1)

    return qas


def get_tbl_context(t1):
    context = ''

    header_tok = t1['header']
    # Here Join scheme can be changed.
    header_joined = ' '.join(header_tok)
    context += header_joined

    return context

def generate_wikisql_bert(path_wikisql, dset_type):
    path_q = os.path.join(path_wikisql, f'{dset_type}.jsonl')
    path_tbl = os.path.join(path_wikisql, f'{dset_type}.tables.jsonl')

    # Generate new json file
    with open(path_tbl, 'r') as f_tbl:
        wikisql = {'version': "v1.1"}
        data = []
        data1 = {}
        paragraphs = []  # new tbls
        for i, t1 in enumerate(f_tbl):
            paragraphs1 = {}

            t1 = json.loads(t1)
            tid = t1['id']
            qas = get_qas(path_q, tid)

            paragraphs1['qas'] = qas
            paragraphs1['tid'] = tid
            paragraphs1['context'] = get_tbl_context(t1)
            #         paragraphs1['context_page_title'] = t1['page_title'] # not always present
            paragraphs1['context_headers'] = t1['header']
            paragraphs1['context_headers_type'] = t1['types']
            paragraphs1['context_contents'] = t1['rows']

            paragraphs.append(paragraphs1)
        data1['paragraphs'] = paragraphs
        data1['title'] = 'wikisql'
        data.append(data1)
        wikisql['data'] = data

    # Save
    with open(os.path.join(path_wikisql, f'{dset_type}_bert.json'), 'w', encoding='utf-8') as fnew:
        json_str = json.dumps(wikisql, ensure_ascii=False)
        json_str += '\n'
        fnew.writelines(json_str)


if __name__=='__main__':

    # 0. Load wikisql
    path_h = '/Users/wonseok'
    path_wikisql = os.path.join(path_h, 'data', 'WikiSQL-1.1', 'data')


    dset_type_list = ['dev', 'test', 'train']

    for dset_type in dset_type_list:
        generate_wikisql_bert(path_wikisql, dset_type)






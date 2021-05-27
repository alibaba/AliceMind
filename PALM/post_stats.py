import argparse
from os import path
from functools import reduce
import re

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def n_grams(tokens, n):
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in range(l) if i + n < l]

def has_repeat(elements):
    d = set(elements)
    return len(d) < len(elements)

def cal_self_repeat(summary):
    ngram_repeats = {2: 0, 4: 0, 8: 0}
    sents = summary.split('<q>')
    for n in ngram_repeats.keys():
        # Respect sentence boundary
        grams = reduce(lambda x, y: x + y, [n_grams(sent.split(), n) for sent in sents], [])
        ngram_repeats[n] += has_repeat(grams)
    return ngram_repeats

def cal_novel(summary, gold, source, summary_ngram_novel, gold_ngram_novel):
    summary = summary.replace('<q>',' ')
    summary = re.sub(r' +', ' ', summary).strip()
    gold = gold.replace('<q>',' ')
    gold = re.sub(r' +', ' ', gold).strip()
    source = source.replace(' ##','')
    source = source.replace('[CLS]',' ').replace('[SEP]',' ').replace('[PAD]',' ')
    source = re.sub(r' +', ' ', source).strip()


    for n in summary_ngram_novel.keys():
        summary_grams = set(n_grams(summary.split(), n))
        gold_grams = set(n_grams(gold.split(), n))
        source_grams = set(n_grams(source.split(), n))
        joint = summary_grams.intersection(source_grams)
        novel = summary_grams - joint
        summary_ngram_novel[n][0] += 1.0*len(novel)
        summary_ngram_novel[n][1] += len(summary_grams)
        summary_ngram_novel[n][2] += 1.0 * len(novel) / (len(summary.split()) + 1e-6)
        joint = gold_grams.intersection(source_grams)
        novel = gold_grams - joint
        gold_ngram_novel[n][0] += 1.0*len(novel)
        gold_ngram_novel[n][1] += len(gold_grams)
        gold_ngram_novel[n][2] += 1.0 * len(novel) / (len(gold.split()) + 1e-6)


def cal_repeat(args):
    candidate_lines = open(args.result_path+'.candidate').read().strip().split('\n')
    gold_lines = open(args.result_path+'.gold').read().strip().split('\n')
    src_lines = open(args.result_path+'.raw_src').read().strip().split('\n')
    lines = zip(candidate_lines,gold_lines,src_lines)

    summary_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}
    gold_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}

    for c,g,s in lines:
        # self_repeats = cal_self_repeat(c)
        cal_novel(c, g, s,summary_ngram_novel, gold_ngram_novel)
    print(summary_ngram_novel, gold_ngram_novel)

    for n in summary_ngram_novel.keys():
        # summary_ngram_novel[n] = summary_ngram_novel[n][2]/len(src_lines)
        # gold_ngram_novel[n] = gold_ngram_novel[n][2]/len(src_lines)
        summary_ngram_novel[n] = summary_ngram_novel[n][0]/summary_ngram_novel[n][1]
        gold_ngram_novel[n] = gold_ngram_novel[n][0]/gold_ngram_novel[n][1]
    print(summary_ngram_novel, gold_ngram_novel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-result_path", default='../../results/cnndm.0')


    args = parser.parse_args()
    eval(args.mode + '(args)')

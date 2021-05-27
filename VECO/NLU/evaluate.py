# coding=utf-8
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation."""

import argparse
from seqeval.metrics import precision_score, recall_score, f1_score
import sys
import os
from collections import defaultdict
import json

from third_party.evaluate_squad import evaluate as squad_eval
from third_party.evaluate_mlqa import evaluate as mlqa_eval

def read_tag(file):
  labels = []
  example = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        example.append(line)
      else:
        labels.append(example)
        example = []
  return labels


def read_label(file):
  with open(file, 'r') as f:
    return [l.strip() for l in f]


def read_squad(file):
  expected_version = '1.1'
  with open(file) as dataset_file:
    dataset_json = json.load(dataset_file)
    # if 'version' in dataset_json and dataset_json['version'] != expected_version:
    #   print('Evaluation expects v-' + expected_version,
    #         ', but got dataset with v-' + dataset_json['version'],
    #         file=sys.stderr)
    if 'data' in dataset_json:
      return dataset_json['data']
    else:
      return dataset_json


def f1(labels, predictions, language=None):
  f1 = f1_score(labels, predictions)
  precision = precision_score(labels, predictions)
  recall = recall_score(labels, predictions)
  return {'f1': f1 * 100, 'precision': precision * 100, 'recall': recall * 100}


def accuracy(labels, predictions, language=None):
  correct = sum([int(p == l) for p, l in zip(predictions, labels)])
  accuracy = float(correct) / len(predictions)
  return {'accuracy': accuracy * 100}

def bucc_f1(labels, predictions, language=None):
  labels = set([tuple(l.split('\t')) for l in labels])
  predictions = set([tuple(l.split('\t')) for l in predictions])
  ncorrect = len(labels.intersection(predictions))
  if ncorrect > 0:
    precision = ncorrect / len(predictions)
    recall = ncorrect / len(labels)
    f1 = 2 * precision * recall / (precision + recall)
  else:
    precision = recall = f1 = 0
  return {'f1': f1 * 100, 'precision': precision * 100, 'recall': recall * 100}

def squad_em_f1(labels, predictions, language=None):
  return squad_eval(labels, predictions)

def mlqa_em_f1(labels, predictions, language):
  if language is None:
    print('required 2-char language code for the argument `language`')
    exit(0)
  return mlqa_eval(labels, predictions, language)


GROUP2TASK = {
  "classification": ["pawsx", "xnli"],
  "tagging": ["udpos", "panx"],
  "qa": ["xquad", "mlqa", "tydiqa"],
  "retrieval": ["bucc2018", "tatoeba"],
}


TASK2LANGS = {
  "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
  "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),
  "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(","),
  "udpos": "af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh".split(","),
  "bucc2018": "de,fr,ru,zh".split(","),
  "tatoeba": "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(","),
  "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),
  "mlqa": "en,es,de,ar,hi,vi,zh".split(","),
  "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),
}


READER_FUNCTION = {
  'pawsx': read_label,
  'xnli': read_label,
  'panx': read_tag,
  'udpos': read_tag,
  'bucc2018': read_label,
  'tatoeba': read_label,
  'xquad': read_squad,
  'mlqa': read_squad,
  'tydiqa': read_squad,
}


METRIC_FUNCTION = {
  'pawsx': accuracy,
  'xnli': accuracy,
  'panx': f1,
  'udpos': f1,
  'bucc2018': bucc_f1,
  'tatoeba': accuracy,
  'xquad': squad_em_f1,
  'mlqa': mlqa_em_f1,
  'tydiqa': squad_em_f1,
}


def evaluate_one_task(prediction_file, label_file, task, language=None):
  """Evalute the classification tasks by accuracy.
  Args:
    prediction_file (string): path to the prediction tsv file.
    label_file (string): path to the grouth truth tsv file.
  Return:
    result (dict): a dictionary with accuracy.

  Both input files contain one example per line as follows:
    ``[label]\t[sentence1]\t[sentence2]``
  """
  predictions = READER_FUNCTION[task](prediction_file)
  labels = READER_FUNCTION[task](label_file)
  if task not in ['bucc2018', 'mlqa', 'tydiqa', 'xquad']:
    assert len(predictions) == len(labels), 'Number of examples in {} and {} not matched in {} task'.format(prediction_file, label_file, task)
  result = METRIC_FUNCTION[task](labels, predictions, language)
  return result


def evaluate(prediction_folder, label_folder, verbose=False):
  """Evaluate on all tasks if available.
  Args:
    prediction_folder (string): prediction folder that contains each task's prediction in each subfolder.
    label_file (string): label folder that contains each task's ground-truth label in each subfolder.
  Return:
    overall_scores (dict): a dictionary with sub-group scores. key: group label.
    detailed_scores (dict): a dictionary with all detailed scores. key: task label.
  """
  prediction_tasks = next(os.walk(prediction_folder))[1]
  label_tasks = next(os.walk(label_folder))[1]
  # prediction_tasks = label_tasks = ['mlqa', 'tydiqa', 'xquad']

  detailed_scores = {}
  for task, langs in TASK2LANGS.items():
    if task in prediction_tasks and task in label_tasks:
      suffix = "json" if task in GROUP2TASK["qa"] else "tsv"
      # collect scores over all languages
      score = defaultdict(dict)
      for lg in langs:
        prediction_file = os.path.join(prediction_folder, task, f"test-{lg}.{suffix}")
        label_file = os.path.join(label_folder, task, f"test-{lg}.{suffix}")
        score_lg = evaluate_one_task(prediction_file, label_file, task, language=lg)
        for metric in score_lg:
          score[metric][lg] = score_lg[metric]
      # average over all languages
      avg_score = {}
      for m in score:
        avg_score[f'avg_{m}'] = sum(score[m].values()) / len(score[m])
      score.update(avg_score)
      if task in GROUP2TASK["qa"]:
        score['avg_metric'] = (score['avg_exact_match'] + score['avg_f1']) / 2
      elif 'avg_f1' in score:
        score['avg_metric'] = score['avg_f1']
      elif 'avg_accuracy' in score:
        score['avg_metric'] = score['avg_accuracy']
      detailed_scores[task] = score
      if verbose:
        avg_result = ', '.join(['{}={:.1f}'.format(k, v) for k, v in score.items() if k.startswith('avg')])
        print('- Evaluate {}:\t{}'.format(task, avg_result))

  # Display logic:
  overall_scores = {}
  all_tasks = set(TASK2LANGS.keys())
  available_tasks = set(detailed_scores.keys())

  # If scores of all tasks are available, show the overall score in the main table
  if all_tasks == available_tasks:
    overall_scores['all_task'] = sum(detailed_scores[task]['avg_metric'] for task in all_tasks) / len(all_tasks)

  # If scores of all tasks in a sub group are available, show the score in the sub table
  for group, group_tasks in GROUP2TASK.items():
    if len(set(group_tasks) - available_tasks) == 0:
      overall_scores[group] = sum(detailed_scores[task]['avg_metric'] for task in group_tasks) / len(group_tasks)

  return overall_scores, detailed_scores


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--prediction_folder', default=None, type=str, required=True,
    help='the predictions of one model')
  parser.add_argument('--label_folder', default=None, type=str, required=True,
    help='the grouth truth file')
  parser.add_argument('--verbose', action='store_true', default=False,
    help='whether to print details')
  args = parser.parse_args()
  overall_scores, detailed_scores = evaluate(args.prediction_folder, args.label_folder, args.verbose)
  overall_scores.update(detailed_scores)
  print(json.dumps(overall_scores))

# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
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

import sofa
sofa.environ("huggingface")
from sofa import run_classification_hf
from sofa import run_sequence_labeling_hf
from sofa import run_generation_hf
import os
import json

try:
    import transformers
    import datasets
    from transformers.utils import check_min_version
    from transformers.utils.versions import require_version

    check_min_version("4.10.0")
    from transformers import HfArgumentParser, TrainingArguments
except Exception as e:
    print(f"Running this script need datasets, transformers>=4.10.0, please check the installation.")
    raise e

if __name__ == "__main__":
    r"""
    This file is used to guide the NLU examples. Supported task types:
    - regression: a regression task(MSE loss).
    - single_label_classification: one label for one or two sentenses(CELoss).
    - multi_label_classification: multiple labels for one or two sentenses(BCELoss).
    - token_classification: NER task, etc.
    For other task types, user can directly use scripts listed at:
    https://github.com/huggingface/transformers/tree/main/examples/pytorch
    or the custom code, just slightly change your code like this:

    ```python
    >>> import sofa
    >>> sofa.environ("huggingface")
    >>> # your original code here, but don't forget to use pretrain_model_name_or_path with our models' ckpt :)
    ```
    """

    # all parameters TrainingArguments are supported.
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        "--pretrain_model_path", default=None, type=str, required=True,
        help="the local path of the pretrained model."
    )
    parser.add_argument(
        "--max_sequence_length", default=None, type=int, required=True,
        help="the max sequence length."
    )
    parser.add_argument(
        "--train_file_path", default=None, type=str, required=False,
        help="the local path of the train file."
    )
    parser.add_argument(
        "--dev_file_path", default=None, type=str, required=False,
        help="the local path of the dev file which is used to cross-validation in training."
    )
    parser.add_argument(
        "--task_type", default=None, type=str, required=True,
        help="the task type, support:regression/single_label_classification/multi_label_classification"
             "/token_classification"
    )
    parser.add_argument(
        "--pair", default=0, type=int, required=False,
        help="1 for pair sentense input, 0 for single sentense input"
    )
    parser.add_argument(
        "--text_a_idx", default=0, type=int, required=False,
        help="Index of text a in dataset."
    )
    parser.add_argument(
        "--text_b_idx", default=1, type=int, required=False,
        help="Index of text b in dataset."
    )
    parser.add_argument(
        "--label_idx", default=-1, type=int, required=False,
        help="Index of label in dataset."
    )
    parser.add_argument(
        "--model_args", default=None, type=str, required=False,
        help="extra model args, pass in as a json string. e.g. '{\"adv_grad_factor\":null}'"
    )
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=False,
        help="Dataset name, which will be passed as first input parameter to function datasets.load_dataset(). \
              You can find the list of datasets on the Hub at https://huggingface.co/datasets."
    )
    parser.add_argument(
        "--cache_dir", default=None, type=str, required=False,
        help="Directory to read/write data. Defaults to \"$PWD/.cache\"."
    )
    parser.add_argument(
        "--train_dataset_config", default=None, type=str, required=False,
        help="train dataset config separated by ','."
    )
    parser.add_argument(
        "--eval_dataset_config", default=None, type=str, required=False,
        help="eval dataset config separated by ','."
    )
    parser.add_argument(
        "--label_enumerate_values", default=None, type=str, required=False,
        help="Pass in a list as the labels. Else the train file will be parsed to get the labels."
    )
    args = parser.parse_args()

    kwargs = {}
    for p in TrainingArguments.__dataclass_fields__.keys():
        if hasattr(args, p):
            kwargs[p] = getattr(args, p)

    if args.task_type == "token_classification":
        # token classification
        run_sequence_labeling_hf(args.pretrain_model_path,
                                 train_file_path=args.train_file_path,
                                 first_sequence=lambda x: x["text"].split("\t")[0].split(" "),
                                 label=lambda x: x["text"].split("\t")[-1].split(" "),
                                 dev_file_path=args.dev_file_path,
                                 config_args={} if args.model_args is None else json.loads(args.model_args),
                                 **kwargs
                                 )
    elif args.task_type == "generation":
        run_generation_hf(args.pretrain_model_path,
                          task_type=args.task_type,
                          train_file_path=args.train_file_path,
                          dev_file_path=args.dev_file_path,
                          config_args={} if args.model_args is None else json.loads(args.model_args),
                          **kwargs
                          )
    else:
        # label classification
        first_sequence = 'sentence' if not args.pair else 'premise'
        second_sequence = None if not args.pair else 'hypothesis'
        label = 'label'
        label_enumerate_values=None if not args.label_enumerate_values else args.label_enumerate_values.split(',')

        if args.train_dataset_config is not None and args.train_dataset_config == "qnli":
            first_sequence = "question"
            second_sequence = "sentence"
        if args.train_dataset_config is not None and args.train_dataset_config == "qqp":
            first_sequence = "question1"
            second_sequence = "question2"

        # label_enumerate_values=None if not args.label_enumerate_values else args.label_enumerate_values.split(',')

        run_classification_hf(pretrain_model_path=args.pretrain_model_path,
                              task_type=args.task_type,
                              first_sequence=first_sequence,
                              second_sequence=second_sequence,
                              label=label,
                              train_file_path=args.train_file_path,
                              dev_file_path=args.dev_file_path,
                              # dataset_name can be a dataset name or a scipt name
                              # here we use the default "text" parser
                              # if input format is json, please change it to "json"
                              dataset_name="text" if not args.dataset_name else args.dataset_name,
                              train_dataset_config=None if not args.train_dataset_config else args.train_dataset_config,
                              eval_dataset_config=None if not args.eval_dataset_config else args.eval_dataset_config,
                              cache_dir=None if not args.cache_dir else args.cache_dir,
                              label_enumerate_values=label_enumerate_values,
                              config_args={} if args.model_args is None else json.loads(args.model_args),
                              **kwargs
                              )

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

import logging
import os
import types
from functools import reduce
import numpy as np

logger = logging.getLogger(__name__)


def run_classification_hf(pretrain_model_path,
                          output_dir,
                          task_type,
                          first_sequence,
                          label,
                          train_file_path=None,
                          dataset_name="text",
                          train_dataset_config=None,
                          eval_dataset_config=None,
                          second_sequence=None,
                          dev_file_path=None,
                          child_tuning_type=None,
                          reserve_p=0.2,
                          label_enumerate_values=None,
                          sequence_length=128,
                          map_function=None,
                          filter_function=None,
                          load_best_model_at_end=True,
                          save_strategy="steps",
                          save_total_limit=1,
                          evaluation_strategy="steps",
                          seed=42,
                          config_args=None,
                          **kwargs,
                          ):
    """
    Run a classsification task with transformers code.
    Support regression/single_label/multi_label tasks
    :param config_args:
    :param pretrain_model_path: The local dir of pretrained model
    :param train_file_path: The local dir or path of train files
    :param task_type: The task type, support:
    - regression: a regression task(MSE loss).
    - single_label_classification: one label for one or two sentenses(CELoss).
    - multi_label_classification: multiple labels for one or two sentenses(BCELoss).
    :param first_sequence: Way to get the first sentense, can be "str" or "FunctionType"
    - str: Will be used in datasets.map(), and used like: examples[first_sequence], this is useful in json files.
    - FunctionType: Will be used in datasets.map(), and used like:first_sequence(examples)
    Examples:
    ``` python
    >>> # way to parse the first sentense out from a tsv-like file
    >>> first_sequence = lambda examples: examples["text"].split("\t")[0]
    >>> # or
    >>> first_sequence = "text"
    ```
    :param label: Way to get the label, can be "str" or "FunctionType"
    - str: Will be used in datasets.map(), and used like: examples[label], this is useful in json files.
    available in "regression" or "single_label_classification"
    - FunctionType: Will be used in datasets.map(), and used like:label(examples)
    available in "regression" or "single_label_classification" or "multi_label_classification"
    If is a regression task, please make sure the return label is a float or a float-like string.
    If is a single_label task, please make sure the return label is a string.
    If is a multi_label task, please make sure the return label is a string or a list.
    Examples
    ``` python
    >>> # Way to parse the multiple labels out from a tsv-like file(Pretend that
    >>> # labels in one column and seperated by a comma)
    >>> label = lambda examples: examples["text"].split("\t")[1].split(",")
    >>> # or
    >>> first_sequence = "text"
    ```
    :param dataset_name: The dataset name passed into datasets.load_dataset. Default will be "text"
    :param second_sequence: Way to get the second sentense, can be "str" or "FunctionType", please follow the rules
    of "first_sequence"
    :param dev_file_path: The local dir of dev file, which is used in cross-validation in training.
    :param child_tuning_type: The child_tuning type. Can be "ChildTuning-F", "ChildTuning-D" or None
    :param reserve_p: the drop-out rate of child_tuning, default 0.2
    :param label_enumerate_values: Pass in a list as the labels. Else the train file will be parsed to get the labels
    :param sequence_length: The max sequence length for padding.
    :param map_function: An optional map function, will be used with datasets.map()
    :param filter_function: An optional filter function, will be used with datasets.filter()
    :param load_best_model_at_end: TrainingArguments.
    :param save_strategy: TrainingArguments.
    :param save_total_limit: TrainingArguments.
    :param evaluation_strategy: TrainingArguments.
    :param seed: Random seed, default 42.
    :param kwargs: Other optional hyper-parameters which is used in TrainingArguments.
    :return: None
    """

    # sofa custom code
    if config_args is None:
        config_args = {}
    from ... import environ
    environ("huggingface")
    # end
    from transformers import TrainingArguments
    from transformers import Trainer, set_seed
    from datasets import load_dataset, interleave_datasets
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # sofa child_tuning
    from ...utils import apply_child_tuning_to_trainer
    import torch

    cache_dir = ".cache"
    if "cache_dir" in kwargs:
        cache_dir = kwargs["cache_dir"]

    set_seed(seed)
    assert type(label) in (str, types.FunctionType)
    logger.info(f"Cuda available:{torch.cuda.is_available()}")

    # prepare data
    train_datasets = None
    dev_datasets = None

    def get_files(files_or_path):
        if type(files_or_path) is str:
            if os.path.isfile(files_or_path):
                return files_or_path
            if os.path.isdir(files_or_path):
                return [os.path.join(files_or_path, f) for f in os.listdir(files_or_path)]
        if type(files_or_path) is list:
            return files_or_path

    if dataset_name != 'text' and train_dataset_config is not None:
        data_sets = []
        for data_config in train_dataset_config.split(','):
            data_sets.append(load_dataset(dataset_name, data_config, split="train", cache_dir=cache_dir))
        train_datasets = interleave_datasets(data_sets)
    elif train_file_path is not None:
        train_files = get_files(train_file_path)
        data_files = {"train": train_files}
        train_datasets = load_dataset(dataset_name, split="train", data_files=data_files)
    if train_datasets is None:
        logger.error(f"dataset_name and train_file_path cannot both be None")

    if dataset_name != 'text' and eval_dataset_config is not None:
        data_sets = []
        for data_config in eval_dataset_config.split(','):
            split_text = "validation" if eval_dataset_config != "mnli" else "validation_matched"
            data_sets.append(load_dataset(dataset_name, data_config, split=split_text,
                                          cache_dir=cache_dir))
        dev_datasets = interleave_datasets(data_sets)
    elif dev_file_path is not None:
        dev_files = get_files(dev_file_path)
        data_files = {"dev": dev_files}
        dev_datasets = load_dataset(dataset_name, split="dev", data_files=data_files)

    if filter_function is not None:
        train_datasets = train_datasets.filter(filter_function)
        if dev_datasets:
            dev_datasets = dev_datasets.filter(filter_function)
    if map_function is not None:
        train_datasets = train_datasets.map(map_function)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_function)

    if task_type == "single_label_classification":
        def map_labels(examples):
            if isinstance(label, str):
                examples["label_map"] = examples[label]
            else:
                examples["label_map"] = label(examples)
            return examples

        train_datasets = train_datasets.map(map_labels)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_labels)

        if label_enumerate_values is None:
            label_enumerate_values = list(set(train_datasets["label_map"]))
            label_enumerate_values.sort()

        id2label = {}
        label2id = {}
        for i in range(len(label_enumerate_values)):
            id2label[i] = label_enumerate_values[i]
            label2id[label_enumerate_values[i]] = i

        model_args = {
            "id2label": id2label,
            "label2id": label2id,
        }

        def map_labels(examples):
            examples["label"] = label2id[examples["label_map"]]
            return examples

        train_datasets = train_datasets.map(map_labels)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_labels)
    elif task_type == "multi_label_classification":
        assert isinstance(label, types.FunctionType)

        def map_labels(examples):
            examples["label_map"] = label(examples)
            return examples

        train_datasets = train_datasets.map(map_labels)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_labels)

        if label_enumerate_values is None:
            label_enumerate_values = list(set(reduce(lambda x, y: (x if isinstance(x, list)
                                                                   else [x]) + (y if isinstance(y, list) else [y]),
                                                     train_datasets["label_map"])))
            label_enumerate_values.sort()

        id2label = {}
        label2id = {}
        for i in range(len(label_enumerate_values)):
            id2label[i] = label_enumerate_values[i]
            label2id[label_enumerate_values[i]] = i

        def label_to_one_hot(examples):
            label_list = examples["label_map"]
            if not isinstance(label_list, list):
                label_list = [label_list]
            labels = [0.0] * len(label_enumerate_values)
            for idx in label_list:
                labels[label2id[idx]] = 1.0
            examples["label"] = labels
            return examples

        train_datasets = train_datasets.map(label_to_one_hot)
        if dev_datasets:
            dev_datasets = dev_datasets.map(label_to_one_hot)
        model_args = {
            "id2label": id2label,
            "label2id": label2id,
        }
    elif task_type == "regression":
        def map_labels(examples):
            if isinstance(label, str):
                examples["label"] = float(examples[label])
            else:
                examples["label"] = float(label(examples))
            return examples

        train_datasets = train_datasets.map(map_labels)
        if dev_datasets:
            dev_datasets = dev_datasets.map(map_labels)
        model_args = {
            "num_labels": 1
        }
    else:
        raise RuntimeError(f"Unsupported task type:{task_type}")

    # Get sbert or veco models/tokenizers
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, model_max_length=sequence_length)
    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_path,
                                                               **model_args,
                                                               **config_args)

    kwargs_for_training = {}
    for p in TrainingArguments.__dataclass_fields__.keys():
        if p in kwargs:
            kwargs_for_training[p] = kwargs[p]
    training_args = TrainingArguments(output_dir=output_dir,
                                      load_best_model_at_end=load_best_model_at_end,
                                      save_strategy=save_strategy,
                                      save_total_limit=save_total_limit,
                                      evaluation_strategy=evaluation_strategy,
                                      **kwargs_for_training,
                                      )

    def tokenize_function(examples):
        assert isinstance(first_sequence, str) or isinstance(first_sequence, types.FunctionType)
        text = examples[first_sequence] if isinstance(first_sequence, str) else first_sequence(examples)
        pair = None if second_sequence is None else examples[second_sequence] \
            if isinstance(second_sequence, str) else second_sequence(examples)
        return tokenizer(text, pair, padding="max_length", truncation=True)

    full_train_dataset = train_datasets.map(tokenize_function)
    if dev_datasets:
        full_eval_dataset = dev_datasets.map(tokenize_function)
    else:
        full_eval_dataset = None

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if task_type == "regression":
            preds = np.squeeze(preds)
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        elif task_type == "single_label_classification":
            preds = np.argmax(preds, axis=1)
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        else:
            from sklearn.metrics import f1_score, precision_score, recall_score

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            probs = sigmoid(preds)
            predictions = (probs > 0.5).astype(int)
            n_class = predictions.shape[1]
            y_trues = np.array(p.label_ids)
            prec_ma = np.sum(np.all(y_trues == predictions, axis=1)) / len(y_trues)
            return {"accuracy": prec_ma}

    trainer = Trainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics
    )
    # apply child_tuning or not.
    if child_tuning_type is not None:
        logger.info("Applying child-tuning.")
        apply_child_tuning_to_trainer(trainer, mode=child_tuning_type, reserve_p=reserve_p)
    trainer.train()

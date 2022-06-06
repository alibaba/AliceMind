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
from typing import Optional, List, Union
import importlib

logger = logging.getLogger(__name__)


def run_sequence_labeling_hf(pretrain_model_path,
                             first_sequence,
                             label,
                             train_file_path,
                             dev_file_path=None,
                             child_tuning_type=None,
                             reserve_p=0.2,
                             label_enumerate_values=None,
                             data_format="text",
                             sequence_length=128,
                             map_function=None,
                             filter_function=None,
                             label_all_tokens=True,
                             load_best_model_at_end=True,
                             save_strategy="steps",
                             save_total_limit=1,
                             evaluation_strategy="steps",
                             return_entity_level_metrics=False,
                             seed=42,
                             config_args=None,
                             **kwargs,
                             ):
    """
        Run a classsification task with transformers code.
        Support regression/single_label/multi_label tasks
        :param config_args:
        :param pretrain_model_path: The local dir of pretrained model
        :param train_file_path: The local dir of train file
        :param first_sequence: Way to get the first sentense, should be "FunctionType"
        Will be used in datasets.map(), and used like:first_sequence(examples)
        Examples:
        ``` python
        >>> # way to parse the first sentense out from a tsv-like file, every word is seperated with a backspace.
        >>> # row is "I have a cat\tO O O S-ANI"
        >>> first_sequence = lambda examples: examples["text"].split("\t")[0].split(" ")
        ```
        :param label: Way to get the label, should be "FunctionType"
        Will be used in datasets.map(), and used like:label(examples)
        Note: Please make sure the labels match the BIO Tags.
        Examples
        ``` python
        >>> # way to parse the labels out from a tsv-like file
        >>> # row is "I have a cat\tO O O S-ANI"
        >>> label = lambda examples: examples["text"].split("\t")[1].split(" ")
        ```
        :param data_format: The data format passed into datasets.load_dataset. Default will be "text"
        :param return_entity_level_metrics: Evaluation will return every single token's metrics
        :param label_all_tokens: When a word is seperated into sub-words, how token will be mapped
        If True, label will be the "I-" token mapped with the "B-" token
        If False, label will be a invalid value(-100)
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
    import torch
    from transformers import TrainingArguments, set_seed
    from transformers import Trainer
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import DataCollatorForTokenClassification
    # child_tuning
    from ...utils import apply_child_tuning_to_trainer

    set_seed(seed)
    logger.info(f"Cuda available:{torch.cuda.is_available()}")

    data_files = {"train": train_file_path}
    if dev_file_path is not None:
        data_files["dev"] = dev_file_path
    raw_datasets = load_dataset(data_format, data_files=data_files)
    if filter_function is not None:
        raw_datasets = raw_datasets.filter(filter_function)
    if map_function is not None:
        raw_datasets = raw_datasets.map(map_function)

    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, model_max_length=sequence_length)

    def sequence_mapping(examples):
        assert isinstance(first_sequence, types.FunctionType)
        assert isinstance(label, types.FunctionType)
        examples["tokenized_sequence"] = first_sequence(examples)
        examples["label_map"] = label(examples)
        return examples

    raw_datasets = raw_datasets.map(sequence_mapping)

    if label_enumerate_values is None:
        label_enumerate_values = list(set(reduce(lambda x, y: x + y, raw_datasets["train"]["label_map"])))
        label_enumerate_values.sort()

    id2label = {}
    label2id = {}
    for i in range(len(label_enumerate_values)):
        id2label[i] = label_enumerate_values[i]
        label2id[label_enumerate_values[i]] = i

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_enumerate_values):
        if label.startswith("B-") and label.replace("B-", "I-") in label_enumerate_values:
            b_to_i_label.append(label_enumerate_values.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["tokenized_sequence"], truncation=True, is_split_into_words=True)

        labels = []
        for idx, label_row in enumerate(examples["label_map"]):
            label_row = [label2id[lb] for lb in label_row]
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_row[word_idx])
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label_row[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    training_args = TrainingArguments(load_best_model_at_end=load_best_model_at_end,
                                      save_strategy=save_strategy,
                                      save_total_limit=save_total_limit,
                                      evaluation_strategy=evaluation_strategy,
                                      **kwargs,
                                      )
    model = AutoModelForTokenClassification.from_pretrained(pretrain_model_path,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            **config_args)

    import numpy as np

    # Code from huggingface datasets.seqeval, some users cannot download the file.
    def _compute(
            predictions,
            references,
            suffix: bool = False,
            scheme: Optional[str] = None,
            mode: Optional[str] = None,
            sample_weight: Optional[List[int]] = None,
            zero_division: Union[str, int] = "warn",
    ):
        from seqeval.metrics import accuracy_score, classification_report
        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")
        report = classification_report(
            y_true=references,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        scores = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in report.items()
        }
        scores["overall_precision"] = overall_score["precision"]
        scores["overall_recall"] = overall_score["recall"]
        scores["overall_f1"] = overall_score["f1-score"]
        scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)
        return scores

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = _compute(predictions=true_predictions, references=true_labels)
        if return_entity_level_metrics:
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    full_train_dataset = tokenized_datasets["train"]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    if dev_file_path is not None:
        full_eval_dataset = tokenized_datasets["dev"]
    else:
        full_eval_dataset = None
    trainer = Trainer(
        model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset,
        data_collator=data_collator, tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    # apply child_tuning
    if child_tuning_type is not None:
        logger.info("Applying child-tuning.")
        apply_child_tuning_to_trainer(trainer, mode=child_tuning_type, reserve_p=reserve_p)
    trainer.train()

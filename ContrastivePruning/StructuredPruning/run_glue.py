#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from torch.utils.data import DataLoader

from model.PruneBert import PruneBertForSequenceClassification
from model.TeacherBert import TeacherBertForSequenceClassification
from prune.prune_utils import determine_pruning_sequence, what_to_prune_head, calculate_head_and_intermediate_importance, what_to_prune_mlp

from transformers.models.bert import BertPreTrainedModel, BertModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from tqdm import tqdm

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class CAPTrainingArguments(TrainingArguments):
    do_prune: bool = field(
        default=False,
    )
    prune_percent: str = field(
        default="",
    )
    retrain_num_train_epochs: float = field(
        default=1,
    )
    ce_loss_weight: float = field(
        default=1,
    )
    cl_unsupervised_loss_weight: float = field(
        default=0.0,
    )
    cl_supervised_loss_weight: float = field(
        default=0.0,
    )
    contrastive_temperature: float = field(
        default=0.1,
    )
    extra_examples: int = field(
        default=4096,
    )
    use_contrastive_loss: bool = field(
        default=False,
    )
    alignrep: str = field(
        default='cls',
    )
    # for structured pruning
    at_least_x_heads_per_layer: int = field(
        default=1,
    )
    normalize_pruning_by_layer: bool = field(
        default=False,
    )
    subset_ratio: float = field(
        default=1.0,
    )
    # for distillation
    use_distill: bool = field(
        default=False,
    )
    teacher_path: str = field(
        default='',
    )
    distill_temperature: float = field(
        default=1.0,
    )
    distill_loss_weight: float = field(
        default=1.0,
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CAPTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue.py", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    teacher = TeacherBertForSequenceClassification.from_pretrained(
        training_args.teacher_path,
        from_tf=bool(".ckpt" in training_args.teacher_path),
        config=config,
        alignrep=training_args.alignrep,
    )
    teacher = teacher.cuda()

    model = PruneBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        contrastive_temperature=training_args.contrastive_temperature,
        ce_loss_weight=training_args.ce_loss_weight,
        cl_unsupervised_loss_weight=training_args.cl_unsupervised_loss_weight,
        cl_supervised_loss_weight=training_args.cl_supervised_loss_weight,
        distill_loss_weight=training_args.distill_loss_weight,
        extra_examples=training_args.extra_examples,
        alignrep=training_args.alignrep,
        get_teacher_logits=teacher.get_teacher_logits if training_args.use_distill else None,
        distill_temperature=training_args.distill_temperature,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
            
        def add_id(examples, idx):
            examples['idx'] = idx
            return examples
        train_dataset = train_dataset.map(
            add_id,
            batched=True,
            with_indices=True, 
        )

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("metric_glue.py", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        global_representations_bank_finetuned = None
        global_representations_bank_pretrained = None
        global_representations_bank_snaps = None

        # encode training examples using fine-tuned model (teacher)
        if training_args.use_contrastive_loss:
            dataloader = DataLoader(
                train_dataset,
                batch_size=trainer.args.train_batch_size,
                shuffle=False,
                collate_fn=trainer.data_collator,
                drop_last=trainer.args.dataloader_drop_last,
                num_workers=trainer.args.dataloader_num_workers,
                pin_memory=trainer.args.dataloader_pin_memory,
            )
            with torch.no_grad():
                for inputs in tqdm(dataloader):
                    inputs = trainer._prepare_inputs(inputs)
                    representations = teacher(encode_example=True, **inputs).cpu()
                    if global_representations_bank_finetuned is None:
                        global_representations_bank_finetuned = representations
                    else:
                        global_representations_bank_finetuned = torch.cat((global_representations_bank_finetuned, representations), dim=0)
            global_representations_bank_finetuned = global_representations_bank_finetuned.unsqueeze(1)

        if not training_args.use_distill:
            teacher = None

        if training_args.do_prune:
            model = trainer._wrap_model(trainer.model, training=False)
            model = model.module if hasattr(model, 'module') else model

            # Determine the number of heads to prune
            prune_percent = training_args.prune_percent
            prune_percent = None if prune_percent == '' else [float(x) for x in prune_percent.split(',')]

            prune_sequence_head, prune_sequence_intermediate = determine_pruning_sequence(
                prune_percent,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.intermediate_size,
                training_args.at_least_x_heads_per_layer,
            )
            prune_sequence = zip(prune_sequence_head, prune_sequence_intermediate)
            
            for step, (n_to_prune_head, n_to_prune_intermediate) in enumerate(prune_sequence):
                logger.info("We are going to prune {} heads and {} intermediate !!!".format(n_to_prune_head, n_to_prune_intermediate))
                head_importance, intermediate_importance = calculate_head_and_intermediate_importance(
                    model,
                    train_dataset,
                    old_head_mask=model.head_mask,
                    old_intermediate_mask=model.intermediate_mask,
                    trainer=trainer,
                    normalize_scores_by_layer=training_args.normalize_pruning_by_layer,
                    subset_size=training_args.subset_ratio
                ) 
                for layer in range(len(head_importance)):
                    layer_scores = head_importance[layer].cpu().data
                    logger.info("head importance score")
                    logger.info("\t".join(f"{x:.5f}" for x in layer_scores))
                # Determine which heads to prune
                new_head_mask = what_to_prune_head(
                    head_importance,
                    n_to_prune=n_to_prune_head,
                    old_head_mask=model.head_mask,
                    at_least_x_heads_per_layer=training_args.at_least_x_heads_per_layer,
                )
                new_intermediate_mask = what_to_prune_mlp(
                    intermediate_importance,
                    n_to_prune=n_to_prune_intermediate,
                    old_intermediate_mask=model.intermediate_mask
                )
                for layer in range(len(new_head_mask)):
                    y = new_head_mask[layer].cpu().data
                    logger.info("head mask")
                    logger.info("\t".join("{}".format(int(x)) for x in y))
                logger.info("intermediate mask")
                for layer in range(len(new_intermediate_mask)):
                    y = new_intermediate_mask[layer]
                    logger.info("Layer {} has {} intermediate active.".format(layer, torch.sum(y)))
                
                # calculate and store example representations and labels (for verification)
                if training_args.use_contrastive_loss:
                    representations_bank = None
                    labels_bank = None
                    dataloader = DataLoader(
                        train_dataset,
                        batch_size=trainer.args.train_batch_size,
                        shuffle=False,
                        collate_fn=trainer.data_collator,
                        drop_last=trainer.args.dataloader_drop_last,
                        num_workers=trainer.args.dataloader_num_workers,
                        pin_memory=trainer.args.dataloader_pin_memory,
                    )
                    with torch.no_grad():
                        for inputs in tqdm(dataloader):
                            inputs = trainer._prepare_inputs(inputs)
                            labels = inputs['labels'].cpu()
                            representations = model(encode_example=True, **inputs).cpu()
                            if representations_bank is None:
                                representations_bank = representations
                                labels_bank = labels
                            else:
                                representations_bank = torch.cat((representations_bank, representations), dim=0)
                                labels_bank = torch.cat((labels_bank, labels), dim=0)

                    if step == 0:
                        # add to global representations bank for pretrained
                        global_representations_bank_pretrained = representations_bank
                        global_representations_bank_pretrained = global_representations_bank_pretrained.unsqueeze(1)
                    else:
                        # add to global representations bank for snaps
                        if global_representations_bank_snaps is None:
                            global_representations_bank_snaps = representations_bank.unsqueeze(1)
                        else:
                            global_representations_bank_snaps = torch.cat((global_representations_bank_snaps, representations_bank.unsqueeze(1)), dim=1)
                        
                    # update bank
                    model.global_representations_bank_finetuned = global_representations_bank_finetuned
                    model.global_representations_bank_pretrained = global_representations_bank_pretrained
                    model.global_representations_bank_snaps = global_representations_bank_snaps
                    model.global_labels_bank = labels_bank

                # apply structured pruing
                model.head_mask[:] = new_head_mask.clone()
                model.intermediate_mask[:] = new_intermediate_mask.clone()
                
                # re-train
                trainer.optimizer = trainer.lr_scheduler = None
                trainer.args.num_train_epochs = training_args.retrain_num_train_epochs
                trainer.train()

                # re-eval
                tasks = [data_args.task_name]
                eval_datasets = [eval_dataset]
                if data_args.task_name == "mnli":
                    tasks.append("mnli-mm")
                    eval_datasets.append(datasets["validation_mismatched"])

                for eval_d, task in zip(eval_datasets, tasks):
                    metrics = trainer.evaluate(eval_dataset=eval_d)
                    max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_d)
                    metrics["eval_samples"] = min(max_val_samples, len(eval_d))
                    new_metrics = {}
                    for k in metrics:
                        new_metrics["{}_{}_{}".format(task, k, step+1)] = metrics[k]
                    metrics = new_metrics
                    trainer.log_metrics("{}_eval_{}".format(task, step+1), metrics)
                    trainer.save_metrics("{}_eval_{}".format(task, step+1), metrics)
        trainer.save_model()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_d, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_d)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_d)
            metrics["eval_samples"] = min(max_val_samples, len(eval_d))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
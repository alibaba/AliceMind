# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import pickle
import modeling
import modeling_labert
import optimization
import tokenization
import tokenization_labert
import tensorflow as tf
import numpy as np
import random
import shutil
from loss_logging_hook import LossLoggingHook
from best_checkpoint_copyer import BestCheckpointCopier

TRAIN_FILE_NAME = "train.txt"
DEV_FILE_NAME = "dev.txt"
TEST_FILE_NAME = "test.txt"
UNLABELED_FILE_NAME = "unlabeled.txt"

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "labert_config_file", None,
    "The config json file corresponding to the pre-trained LaBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the LaBERT model was trained on.")

flags.DEFINE_string("lexicon_file", None,
                    "The lexicon file that the LaBERT model was trained on.")

flags.DEFINE_boolean(
    "use_named_lexicon", False,
    "The lexicon file is named (say in the format of {entry}\t{name}).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained LaBERT model).")

flags.DEFINE_string(
    "label_file", None,
    "The pickle of labels.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "lr_layer_decay_rate", 1.0,
    "Top layer: lr[L] = FLAGS.learning_rate. Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_float("adam_beta1", 0.9, "The beta1 for adam.")
flags.DEFINE_float("adam_beta2", 0.999, "The beta2 for adam.")
flags.DEFINE_float("adam_epsilon", 1e-6, "The epsilon for adam.")

flags.DEFINE_bool("use_as_feature", False, "Specific to use bert as feature.")

flags.DEFINE_string("predict_output", "test_results.tsv",
                    "predict_output file name")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  batches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               positional_embeddings_start,
               positional_embeddings_end,
               label_id,
               is_real_example=True,):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.positional_embeddings_start = positional_embeddings_start
    self.positional_embeddings_end = positional_embeddings_end
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self, data_dir):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def set_labels(self, labels):
    assert isinstance(labels, list)
    self._labels = labels

  def has_train_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, TRAIN_FILE_NAME))

  def has_dev_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, DEV_FILE_NAME))

  def has_test_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, TEST_FILE_NAME))

  def has_unlabeled_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, UNLABELED_FILE_NAME))


def _read_tsv(input_file, quotechar=None):
  """Reads a tab separated value file."""
  with tf.io.gfile.GFile(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for line in reader:
      lines.append(line)
    return lines


class SentenceClassificationProcessor(DataProcessor):
  def __init__(self, tokenizer=None):
    super(SentenceClassificationProcessor, self).__init__()
    self.text_a_id = 0
    self.label_id = 1
    self._labels = None
    self.tokenizer = tokenizer

  def get_train_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, TRAIN_FILE_NAME)), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_unlabeled_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, UNLABELED_FILE_NAME)), "unlabeled")

  def get_labels(self, data_dir):
    lines = _read_tsv(os.path.join(data_dir, TRAIN_FILE_NAME))
    labels = set()
    for index, line in enumerate(lines):
      if len(line) <= self.label_id:
        tf.compat.v1.logging.warn('Line {}: in illegal format, skipped'.format(index))
        continue
      label = tokenization.convert_to_unicode(line[self.label_id])
      labels.add(label)
    self._labels = [label for label in sorted(labels)]
    return self._labels

  def _create_examples(self, lines, set_type):
    examples = []
    for (index, line) in enumerate(lines):
      # skip header
      if len(line) <= self.label_id:
        tf.compat.v1.logging.warn('Line {}: in illegal format, skipped'.format(index))
        continue
      guid = "%s-%s" % (set_type, index)
      if set_type == "test" or set_type == "unlabeled":
        text_a = tokenization.convert_to_unicode(line[self.text_a_id])
        label = None
      else:
        text_a = tokenization.convert_to_unicode(line[self.text_a_id])
        label = tokenization.convert_to_unicode(line[self.label_id])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def get_num_token_stats(self, data_dir):
    examples = []
    if self.has_train_file(data_dir):
      examples.extend(self.get_train_examples(data_dir))
    if self.has_dev_file(data_dir):
      examples.extend(self.get_dev_examples(data_dir))
    if self.has_test_file(data_dir):
      examples.extend(self.get_test_examples(data_dir))
    numbers = []
    for example in examples:
      if self.tokenizer:
        num_tokens = len(self.tokenizer.tokenize(example.text_a)) + 2
      else:
        num_tokens = len(tokenization.convert_to_unicode(example.text_a)) + 2
      numbers.append(num_tokens)

    numbers = np.array(numbers)
    numbers.sort()
    token_stats = {
      "ave": np.mean(numbers),
      "median": np.median(numbers),
      "top80": np.percentile(numbers, 80),
      "top90": np.percentile(numbers, 90),
      "top95": np.percentile(numbers, 95),
      "top99": np.percentile(numbers, 99)
    }
    return token_stats


class SentencePairClassificationProcessor(DataProcessor):
  def __init__(self, tokenizer=None):
    super(SentencePairClassificationProcessor, self).__init__()
    self.text_a_id = 0
    self.text_b_id = 1
    self.label_id = 2
    self._labels = None
    self.tokenizer = tokenizer

  def get_train_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, TRAIN_FILE_NAME)), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, DEV_FILE_NAME)), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, TEST_FILE_NAME)), "test")

  def get_unlabeled_examples(self, data_dir):
    return self._create_examples(_read_tsv(os.path.join(data_dir, UNLABELED_FILE_NAME)), "unlabeled")

  def get_labels(self, data_dir):
    lines = _read_tsv(os.path.join(data_dir, TRAIN_FILE_NAME))
    labels = set()
    for index, line in enumerate(lines):
      if len(line) <= self.label_id:
        tf.compat.v1.logging.warn('Line {}: in illegal format, skipped'.format(index))
        continue
      label = tokenization.convert_to_unicode(line[self.label_id])
      labels.add(label)
    self._labels = [label for label in sorted(labels)]
    return self._labels

  def _create_examples(self, lines, set_type):
    examples = []
    for index, line in enumerate(lines):
      # skip header
      guid = "%s-%s" % (set_type, index)
      if len(line) <= self.label_id:
        tf.compat.v1.logging.warn('Line {}: in illegal format, skipped'.format(index))
        continue
      if set_type == "test" or set_type == "unlabeled":
        text_a = tokenization.convert_to_unicode(line[self.text_a_id])
        text_b = tokenization.convert_to_unicode(line[self.text_b_id])
        label = None
      else:
        text_a = tokenization.convert_to_unicode(line[self.text_a_id])
        text_b = tokenization.convert_to_unicode(line[self.text_b_id])
        label = tokenization.convert_to_unicode(line[self.label_id])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_num_token_stats(self, data_dir):
    examples = []
    if self.has_train_file(data_dir):
      examples.extend(self.get_train_examples(data_dir))
    if self.has_dev_file(data_dir):
      examples.extend(self.get_dev_examples(data_dir))
    if self.has_test_file(data_dir):
      examples.extend(self.get_test_examples(data_dir))
    numbers = []
    for example in examples:
      if self.tokenizer:
        num_tokens = len(self.tokenizer.tokenize(example.text_a)) + \
                     len(self.tokenizer.tokenize(example.text_b)) + 3
      else:
        num_tokens = len(tokenization.convert_to_unicode(example.text_a)) + \
                     len(tokenization.convert_to_unicode(example.text_b)) + 3
      numbers.append(num_tokens)

    numbers = np.array(numbers)
    numbers.sort()
    token_stats = {
      "ave": np.mean(numbers),
      "median": np.median(numbers),
      "top80": np.percentile(numbers, 80),
      "top90": np.percentile(numbers, 90),
      "top95": np.percentile(numbers, 95),
      "top99": np.percentile(numbers, 99)
    }
    return token_stats


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, rng):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
      input_ids=[0] * max_seq_length,
      input_mask=[0] * max_seq_length,
      segment_ids=[0] * max_seq_length,
      positional_embeddings_start=[0] * max_seq_length,
      positional_embeddings_end=[0] * max_seq_length,
      label_id=0,
      is_real_example=False,)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  encoding_a = tokenizer.tokenize(example.text_a, add_candidate_indices=False)
  encoding_b = None
  if example.text_b:
    encoding_b = tokenizer.tokenize(example.text_b, add_candidate_indices=False)

  if encoding_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(encoding_a, encoding_b, max_seq_length - 3, rng)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    while encoding_a.lazy_length() > max_seq_length - 2:
      encoding_a.lazy_pop_back()
    encoding_a.finalize_lazy_pop_back()

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  encoding = tokenizer.build_cls_encoding(add_candidate_indices=False)
  segment_ids = [0]

  encoding.extend(encoding_a)
  segment_ids.extend([0] * len(encoding_a.tokens))

  encoding.extend(tokenizer.build_sep_encoding(add_candidate_indices=False))
  segment_ids.append(0)

  if encoding_b:
    encoding.extend(encoding_b)
    segment_ids.extend([1] * len(encoding_b.tokens))

    encoding.extend(tokenizer.build_sep_encoding(add_candidate_indices=False))
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(encoding.tokens)
  positional_embeddings_start, positional_embeddings_end = encoding.position_embedding(['start', 'end'])

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    positional_embeddings_start.append(0)
    positional_embeddings_end.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(positional_embeddings_start) == max_seq_length
  assert len(positional_embeddings_end) == max_seq_length

  if example.label:
    label_id = label_map[example.label]
  else:
    label_id = 0
  if ex_index < 5:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("guid: %s" % (example.guid))
    tf.compat.v1.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in encoding.tokens]))
    tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.compat.v1.logging.info(
      "positional_embeddings_start: %s" % " ".join([str(x) for x in positional_embeddings_start]))
    tf.compat.v1.logging.info(
      "positional_embeddings_end: %s" % " ".join([str(x) for x in positional_embeddings_end]))
    tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_id=label_id,
    positional_embeddings_start=positional_embeddings_start,
    positional_embeddings_end=positional_embeddings_end,
    is_real_example=True,)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, rng,):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, rng)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["positional_embeddings_start"] = create_int_feature(feature.positional_embeddings_start)
    features["positional_embeddings_end"] = create_int_feature(feature.positional_embeddings_end)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
      [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, add_vis_matrix=False):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
    "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    "positional_embeddings_start": tf.io.FixedLenFeature([seq_length], tf.int64),
    "positional_embeddings_end": tf.io.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.io.FixedLenFeature([], tf.int64),
    "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }
  if add_vis_matrix:
    name_to_features['visibility_matrix'] = tf.io.SparseFeature(
      index_key=['visibility_matrix_i', 'visibility_matrix_j'],
      value_key='visibility_matrix_values',
      dtype=tf.int64, size=[seq_length, seq_length])

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
      tf.contrib.data.map_and_batch(
        lambda record: _decode_record(record, name_to_features),
        batch_size=batch_size,
        drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(encoding_a, encoding_b, max_num_tokens, rng):
  while True:
    total_length = encoding_a.lazy_length() + encoding_b.lazy_length()
    if total_length <= max_num_tokens:
      break

    if encoding_a.lazy_length() > encoding_b.lazy_length():
      trunc_tokens = encoding_a
    else:
      trunc_tokens = encoding_b

    assert trunc_tokens.lazy_length() >= 1

    if rng.random() < 0.5:
      trunc_tokens.lazy_pop_front()
    else:
      trunc_tokens.lazy_pop_back()

  encoding_a.finalize_lazy_pop_front()
  encoding_a.finalize_lazy_pop_back()
  encoding_b.finalize_lazy_pop_front()
  encoding_b.finalize_lazy_pop_back()


def create_model(labert_config, is_training, input_ids, input_mask, segment_ids,
                 positional_embeddings_start, positional_embeddings_end, labels,
                 num_labels, use_one_hot_embeddings, use_as_feature,):
  """Creates a classification model."""
  model = modeling_labert.LaBertModel(
    config=labert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    start_positions=positional_embeddings_start,
    end_positions=positional_embeddings_end,
    use_one_hot_embeddings=use_one_hot_embeddings,
    compute_type=tf.float32,)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value

  if use_as_feature:
    output_layer = tf.stop_gradient(output_layer)
    project_weights = tf.get_variable(
      "project_weights", [hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    project_bias = tf.get_variable(
      "project_bias", [hidden_size], initializer=tf.zeros_initializer())
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.matmul(output_layer, project_weights, transpose_b=True)
    output_layer = tf.nn.bias_add(output_layer, project_bias)
    output_layer = tf.nn.relu(output_layer)

  output_weights = tf.get_variable(
    "output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    total_loss = tf.reduce_mean(per_example_loss)

    return (total_loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, beta1, beta2, epsilon,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, use_as_feature,):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    do_log_information = (FLAGS.do_train and mode == tf.estimator.ModeKeys.TRAIN) or \
                         (not FLAGS.do_train and FLAGS.do_eval and mode == tf.estimator.ModeKeys.EVAL) or \
                         (FLAGS.do_predict and mode == tf.estimator.ModeKeys.PREDICT)

    if do_log_information:
      tf.compat.v1.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    positional_embeddings_start = features["positional_embeddings_start"]
    positional_embeddings_end = features["positional_embeddings_end"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
      bert_config, is_training, input_ids, input_mask, segment_ids,
      positional_embeddings_start, positional_embeddings_end, label_ids,
      num_labels, use_one_hot_embeddings, use_as_feature,)

    total_loss = tf.identity(total_loss, name='total_loss')

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if do_log_information:
      tf.compat.v1.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
        total_loss, learning_rate, beta1, beta2, epsilon, num_train_steps, num_warmup_steps,
        None, False)

      logging_hook = LossLoggingHook(params['batch_size'], every_n_iter=int(num_train_steps / 200 + 1))

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(labels=label_ids, predictions=predictions)
        loss = tf.compat.v1.metrics.mean(values=per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, rng):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, rng)

    features.append(feature)
  return features


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
      "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  labert_config = modeling_labert.LaBertConfig.from_json_file(FLAGS.labert_config_file)

  if FLAGS.max_seq_length > labert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, labert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in ("single", "pair"):
    raise ValueError("Task not found: %s" % (task_name))

  if task_name == "single":
    processor = SentenceClassificationProcessor()
  else:
    processor = SentencePairClassificationProcessor()

  tpu_cluster_resolver = None

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  save_checkpoints_steps = 1000
  if FLAGS.do_train:
    label_list = processor.get_labels(FLAGS.data_dir)
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    save_checkpoints_steps = num_train_steps // 10
    if save_checkpoints_steps == 0:
      save_checkpoints_steps = num_train_steps
  else:
    label_list = pickle.load(open(FLAGS.label_file, 'rb'))
    processor.set_labels(label_list)

  if FLAGS.use_named_lexicon:
    tokenizer = tokenization_labert.LatticeTokenizerWithMapping(
      vocab_file=FLAGS.vocab_file,
      lexicon_file=FLAGS.lexicon_file,
      do_lower_case=FLAGS.do_lower_case)
  else:
    tokenizer = tokenization_labert.LatticeTokenizer(
      vocab_file=FLAGS.vocab_file,
      lexicon_file=FLAGS.lexicon_file,
      do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    keep_checkpoint_max=1,
    log_step_count_steps=1 << 25,
    tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=8,
      per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
    bert_config=labert_config,
    num_labels=len(label_list),
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    beta1=FLAGS.adam_beta1,
    beta2=FLAGS.adam_beta2,
    epsilon=FLAGS.adam_epsilon,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_one_hot_embeddings=False,
    use_as_feature=FLAGS.use_as_feature,)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  train_input_fn = None
  rng = random.Random(FLAGS.random_seed)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, rng)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=False,)

  eval_input_fn = None
  if FLAGS.do_eval and processor.has_dev_file(FLAGS.data_dir):
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, rng,)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(eval_examples), num_actual_eval_examples,
                              len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,)

  if FLAGS.do_train and FLAGS.do_eval and processor.has_dev_file(FLAGS.data_dir):
    best_ckpt_exporter = BestCheckpointCopier(name='best', checkpoints_to_keep=1,
                                              score_metric='eval_accuracy',
                                              compare_fn=lambda x, y: x.score > y.score,
                                              sort_key_fn=lambda x: x.score,
                                              sort_reverse=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      throttle_secs=0,
                                      exporters=[best_ckpt_exporter],
                                      steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

    with open(os.path.join(FLAGS.output_dir, 'labels.pkl'), 'wb') as writer:
      pickle.dump(label_list, file=writer, protocol=3)

    shutil.copy(FLAGS.vocab_file, os.path.join(FLAGS.output_dir, "vocab.txt"))
    shutil.copy(FLAGS.lexicon_file, os.path.join(FLAGS.output_dir, "lexicon.txt"))
    shutil.copy(FLAGS.labert_config_file, os.path.join(FLAGS.output_dir, "labert_config.json"))

  elif FLAGS.do_train:
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    with open(os.path.join(FLAGS.output_dir, 'labels.pkl'), 'wb') as writer:
      pickle.dump(label_list, file=writer, protocol=3)

    shutil.copy(FLAGS.vocab_file, os.path.join(FLAGS.output_dir, "vocab.txt"))
    shutil.copy(FLAGS.lexicon_file, os.path.join(FLAGS.output_dir, "lexicon.txt"))
    shutil.copy(FLAGS.labert_config_file, os.path.join(FLAGS.output_dir, "labert_config.json"))

  elif FLAGS.do_eval and processor.has_dev_file(FLAGS.data_dir):
    result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file, rng,)

    tf.compat.v1.logging.info("***** Running prediction*****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(predict_examples), num_actual_predict_examples,
                              len(predict_examples) - num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False,)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.predict_output)
    with tf.io.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      writer.write("\t".join(label_list) + "\n")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("lexicon_file")
  flags.mark_flag_as_required("labert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run()

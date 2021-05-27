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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import modeling
import modeling_labert
import optimization
import tokenization
import tokenization_labert
import tensorflow as tf
import numpy as np
import tf_metrics
import random
import shutil
import collections
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

flags.DEFINE_bool("do_adversarial_train", False, "Do adversarial training [SMART algorithm].")

flags.DEFINE_string("predict_output", "test_results.tsv",
                    "predict_output file name")


class InputExample(object):

  def __init__(self, guid, words, labels=None):
    self.guid = guid
    self.words = words
    self.labels = labels


class PaddingInputExample(object):
  """ """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               positional_embeddings_start,
               positional_embeddings_end,
               label_positions,
               label_ids,
               label_weights):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.positional_embeddings_start = positional_embeddings_start
    self.positional_embeddings_end = positional_embeddings_end
    self.label_positions = label_positions
    self.label_ids = label_ids
    self.label_weights = label_weights


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def stable_ce_kl(logit, target, epsilon=1e-6):
  # stable kl:
  # Input shape:
  #   logit: [d1, d2, ..., num_labels]
  #   target: [d1, d2, .... num_labels]
  #
  # Output shape:
  #   [d1 * d2 * ..., 1]
  logit = tf.reshape(logit, (-1, logit.shape[-1]))
  target = tf.reshape(target, (-1, target.shape[-1]))
  p = tf.math.exp(tf.math.log_softmax(logit, 1))
  y = tf.math.exp(tf.math.log_softmax(target, 1))
  rp = -tf.math.log((1.0 / (p + epsilon) - 1 + epsilon))
  ry = -tf.math.log((1.0 / (y + epsilon) - 1 + epsilon))
  return tf.reduce_mean((p * (rp - ry) * 2), axis=-1)


def sym_ce_kl_loss(logit, target):
  # sym_kl_loss:
  # Input shape:
  #   logit: [d1, d2, ..., num_labels]
  #   target: [d1, d2, .... num_labels]
  #
  # Output shape:
  #   [d1 * d2 * ..., 1]
  loss = tf.reduce_mean(
    tf.keras.losses.kld(tf.math.log_softmax(logit, axis=-1), tf.math.softmax(target, axis=-1)) + \
    tf.keras.losses.kld(tf.math.log_softmax(target, axis=-1), tf.math.softmax(logit, axis=-1)),
    axis=-1
  )
  return loss


def compute_adv_loss(embedding_output, labert_config, input_ids, input_mask,
                     start_positions, end_positions,
                     num_labels,
                     label_positions, label_weights, is_training,
                     target_logits, noise_epsilon, step_size):
  z = tf.random.normal(tf.shape(embedding_output)) * noise_epsilon

  with tf.compat.v1.variable_scope("bert", reuse=True):
    with tf.compat.v1.variable_scope("embeddings"):
      adv_embedding_output = embedding_output + z

    with tf.compat.v1.variable_scope("encoder"):
      attention_mask = modeling.create_attention_mask_from_input_mask(
        input_ids, input_mask)
      all_encoder_layers = modeling_labert.transformer_model(
        position_embeddings_ids=[start_positions, end_positions],
        input_tensor=adv_embedding_output,
        attention_mask=attention_mask,
        hidden_size=labert_config.hidden_size,
        embedding_size=labert_config.embedding_size,
        num_hidden_layers=labert_config.num_hidden_layers,
        num_attention_heads=labert_config.num_attention_heads,
        intermediate_size=labert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(labert_config.hidden_act),
        hidden_dropout_prob=labert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=labert_config.attention_probs_dropout_prob,
        initializer_range=labert_config.initializer_range,
        do_share_parameter_across_layers=False,
        do_return_all_layers=True,
        do_return_attention_maps=False,
        compute_type=tf.float32)

    adv_output_layer = tf.cast(all_encoder_layers[-1], tf.float32)
    adv_output_layer = gather_indexes(adv_output_layer, label_positions)

  hidden_size = adv_output_layer.shape[-1].value

  root_scope = tf.compat.v1.get_variable_scope()
  with tf.compat.v1.variable_scope(root_scope, reuse=True):
    output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("loss", reuse=True):
    if is_training:
      adv_output_layer = tf.nn.dropout(adv_output_layer, rate=0.1)

    adv_logits = tf.matmul(adv_output_layer, output_weights, transpose_b=True)
    adv_logits = tf.nn.bias_add(adv_logits, output_bias)

  label_weights = tf.reshape(label_weights, [-1])
  adv_loss = stable_ce_kl(adv_logits, tf.stop_gradient(target_logits))
  adv_loss = tf.reshape(adv_loss, [-1])
  adv_loss = tf.reduce_sum(adv_loss * label_weights) / tf.reduce_sum(label_weights)
  delta_grad = tf.compat.v1.gradients(adv_loss, adv_embedding_output)[0]
  norm = tf.norm(delta_grad)
  is_corrupted = tf.math.logical_or(tf.math.is_inf(norm), tf.math.is_nan(norm))

  delta_grad = delta_grad / (tf.math.reduce_max(tf.math.abs(delta_grad), axis=-1, keepdims=True) + 1e-6)

  with tf.compat.v1.variable_scope("bert", reuse=True):
    with tf.compat.v1.variable_scope("embeddings"):
      adv_embedding_output2 = embedding_output + tf.stop_gradient(delta_grad * step_size)

    with tf.compat.v1.variable_scope("encoder"):
      all_encoder_layers2 = modeling_labert.transformer_model(
        input_tensor=adv_embedding_output2,
        attention_mask=attention_mask,
        position_embeddings_ids=[start_positions, end_positions],
        hidden_size=labert_config.hidden_size,
        embedding_size=labert_config.embedding_size,
        num_hidden_layers=labert_config.num_hidden_layers,
        num_attention_heads=labert_config.num_attention_heads,
        intermediate_size=labert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(labert_config.hidden_act),
        hidden_dropout_prob=labert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=labert_config.attention_probs_dropout_prob,
        initializer_range=labert_config.initializer_range,
        do_share_parameter_across_layers=False,
        do_return_all_layers=True,
        do_return_attention_maps=False,
        compute_type=tf.float32)

    adv_output_layer2 = tf.cast(all_encoder_layers2[-1], tf.float32)
    adv_output_layer2 = gather_indexes(adv_output_layer2, label_positions)

  with tf.compat.v1.variable_scope("loss", reuse=True):
    if is_training:
      adv_output_layer2 = tf.nn.dropout(adv_output_layer2, rate=0.1)

    adv_logits2 = tf.matmul(adv_output_layer2, output_weights, transpose_b=True)
    adv_logits2 = tf.nn.bias_add(adv_logits2, output_bias)

  adv_loss2 = sym_ce_kl_loss(adv_logits2, target_logits)
  adv_loss2 = tf.reshape(adv_loss2, [-1])
  adv_loss2 = tf.reduce_sum(adv_loss2 * label_weights) / tf.reduce_sum(label_weights)
  return tf.cond(is_corrupted, lambda: tf.constant(0.), lambda: adv_loss2)


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_unlabeled_examples(self, data_dir):
    raise NotImplementedError()

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

  def get_num_token_stats(self, data_dir):
    raise NotImplementedError()

  def has_train_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, TRAIN_FILE_NAME))

  def has_dev_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, DEV_FILE_NAME))

  def has_test_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, TEST_FILE_NAME))

  def has_unlabeled_file(self, data_dir):
    return os.path.exists(os.path.join(data_dir, UNLABELED_FILE_NAME))


class SequenceLabelingProcessor(DataProcessor):
  def __init__(self, tokenizer=None):
    self._labels = None
    self.tokenizer = tokenizer

  @staticmethod
  def get_raw_data(input_file):
    """Reads a BIO data. block-wise"""
    examples = []
    with tf.io.gfile.GFile(input_file, 'r') as f:
      words, labels = [], []
      for line in f:
        line = line.strip()
        if len(line) == 0:
          if len(words) > 0:
            examples.append((words, labels))
          words, labels = [], []
        else:
          fields = line.strip().split()
          # If the inputs are a list of words -- it's OK
          # If the inputs are a list of word, label pair -- it's also OK.
          # test/unlabeled data is handled by _create_examples
          word, label = fields[0], fields[-1]
          word = tokenization.convert_to_unicode(word)
          label = tokenization.convert_to_unicode(label)
          words.append(word)
          labels.append(label)
    if len(words) > 0:
      examples.append((words, labels))
    return examples

  def get_train_examples(self, data_dir):
    return self._create_examples(
        self.get_raw_data(os.path.join(data_dir, TRAIN_FILE_NAME)), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self.get_raw_data(os.path.join(data_dir, DEV_FILE_NAME)), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(
        self.get_raw_data(os.path.join(data_dir, TEST_FILE_NAME)), "test")

  def get_unlabeled_examples(self, data_dir):
    return self._create_examples(
      self.get_raw_data(os.path.join(data_dir, UNLABELED_FILE_NAME)), "unlabeled")

  def _create_examples(self, raw_data, set_type):
    examples = []
    for (i, (words, labels)) in enumerate(raw_data):
      guid = "%s-%s" % (set_type, i)
      if set_type == "test" or set_type == "unlabeled":
        labels = None
      examples.append(InputExample(guid=guid, words=words, labels=labels))
    return examples

  def get_labels(self, data_dir):
    examples = self.get_train_examples(data_dir)
    labels_set = set()
    for example in examples:
      labels_set.update(example.labels)
    self._labels = [label for label in sorted(labels_set)]
    return self._labels

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
      length = 2
      for word in example.words:
        if self.tokenizer:
          length += len(self.tokenizer.tokenize(word))
        else:
          length += len(tokenization.convert_to_unicode(word))
      numbers.append(length)

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
                           tokenizer):
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        positional_embeddings_start=[0] * max_seq_length,
        positional_embeddings_end=[0] * max_seq_length,
        label_positions=[0] * max_seq_length,
        label_ids=[0] * max_seq_length,
        label_weights=[0.] * max_seq_length)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  segment_ids, label_positions, label_ids, label_weights = [], [], [], []
  segment_ids.append(0)
  position = 1

  encoding = tokenizer.build_cls_encoding(add_candidate_indices=False)
  covered = {(0, 1)}  # The CLS token
  for i, word in enumerate(example.words):
    if example.labels is not None:
      label_id = label_map[example.labels[i]]
    else:
      label_id = 0  # dummy id.

    encoding_of_a_word = tokenizer.tokenize(word, add_candidate_indices=False)
    last_position = encoding.positions[-1] + encoding.lengths[-1]

    for start_position, length in zip(encoding_of_a_word.positions,
                                      encoding_of_a_word.lengths):
      covered.add((last_position + start_position,
                   last_position + start_position + length))

    encoding.extend(encoding_of_a_word)
    for j, _ in enumerate(encoding_of_a_word.tokens):
      if j == 0:
        label_positions.append(position)
        label_ids.append(label_id)
        label_weights.append(1.)

      segment_ids.append(0)
      position += 1

  encoding1 = tokenizer.build_cls_encoding(add_candidate_indices=False)
  encoding1.extend(tokenizer.tokenize("".join(example.words), add_candidate_indices=False))
  for token, start_position, length in zip(encoding1.tokens,
                                           encoding1.positions,
                                           encoding1.lengths):
    if (start_position, start_position + length) in covered:
      continue

    segment_ids.append(0)
    encoding.tokens.append(token)
    encoding.positions.append(start_position)
    encoding.lengths.append(length)

  encoding.extend(tokenizer.build_sep_encoding(add_candidate_indices=False))
  segment_ids.append(0)

  positional_embeddings_start, positional_embeddings_end = encoding.position_embedding(modes=['start', 'end'])

  input_ids = tokenizer.convert_tokens_to_ids(encoding.tokens)

  input_mask = [1] * len(input_ids)

  if len(input_ids) > max_seq_length:
    input_ids = input_ids[:max_seq_length]
    input_mask = input_mask[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    positional_embeddings_start = positional_embeddings_start[:max_seq_length]
    positional_embeddings_end = positional_embeddings_end[:max_seq_length]

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    positional_embeddings_start.append(0)
    positional_embeddings_end.append(0)

  if len(label_positions) > max_seq_length:
    label_positions = label_positions[:max_seq_length]
    label_ids = label_ids[:max_seq_length]
    label_weights = label_weights[:max_seq_length]

  while len(label_positions) < max_seq_length:
    label_positions.append(0)  # it's label padding, not related with padding
    label_ids.append(0)
    label_weights.append(0.)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(positional_embeddings_start) == max_seq_length
  assert len(positional_embeddings_end) == max_seq_length

  if ex_index < 3:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("guid: %s" % example.guid)
    tf.compat.v1.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in encoding.tokens]))
    tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.compat.v1.logging.info(
      "positional_embeddings_start: %s" % " ".join([str(x) for x in positional_embeddings_start]))
    tf.compat.v1.logging.info(
      "positional_embeddings_end: %s" % " ".join([str(x) for x in positional_embeddings_end]))
    tf.compat.v1.logging.info("label_positions: %s" % " ".join([str(x) for x in label_positions]))
    tf.compat.v1.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    tf.compat.v1.logging.info("label_weights: %s" % " ".join([str(x) for x in label_weights]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      positional_embeddings_start=positional_embeddings_start,
      positional_embeddings_end=positional_embeddings_end,
      label_positions=label_positions,
      label_ids=label_ids,
      label_weights=label_weights)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file,):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer,)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    if feature is not None:
      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["positional_embeddings_start"] = create_int_feature(feature.positional_embeddings_start)
      features["positional_embeddings_end"] = create_int_feature(feature.positional_embeddings_end)
      features["label_positions"] = create_int_feature(feature.label_positions)
      features["label_ids"] = create_int_feature(feature.label_ids)
      features["label_weights"] = create_float_feature(feature.label_weights)

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_training_instances=100):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "positional_embeddings_start": tf.io.FixedLenFeature([seq_length], tf.int64),
      "positional_embeddings_end": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_positions": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_weights": tf.io.FixedLenFeature([seq_length], tf.float32),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
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
      d = d.shuffle(min(num_training_instances, 10000))

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(labert_config, is_training, input_ids, input_mask, segment_ids,
                 positional_embeddings_start, positional_embeddings_end,
                 label_positions, num_labels,
                 use_fp16=False, do_return_model=False):
  """Creates a classification model."""
  model = modeling_labert.LaBertModel(
      scope="bert",
      config=labert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      start_positions=positional_embeddings_start,
      end_positions=positional_embeddings_end,
      use_one_hot_embeddings=False,
      compute_type=tf.float16 if use_fp16 else tf.float32,)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_sequence_output()
  hidden_size = output_layer.shape[-1].value

  output_layer = gather_indexes(output_layer, label_positions)

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, rate=0.1)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if do_return_model:
      return logits, model
    return logits


def model_fn_builder(labert_config, num_labels, init_checkpoint,
                     beta1, beta2, epsilon,
                     num_train_steps, lr_layer_decay_rate, pos_indices=None,
                     learning_rate=None, num_warmup_steps=None):
  """Returns `model_fn` closure for TPUEstimator."""
  def _model_fn(features, labels, mode, params):
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
    positional_embeddings_start = features["positional_embeddings_start"]
    positional_embeddings_end = features["positional_embeddings_end"]
    label_positions = features["label_positions"]
    label_ids = features["label_ids"]
    label_weights = features["label_weights"]
    seq_length = label_weights.shape[1]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    payload = create_model(
      labert_config=labert_config, is_training=is_training,
      input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
      positional_embeddings_start=positional_embeddings_start,
      positional_embeddings_end=positional_embeddings_end,
      label_positions=label_positions, num_labels=num_labels,
      do_return_model=FLAGS.do_adversarial_train and is_training)

    if FLAGS.do_adversarial_train and is_training:
      logits, model = payload
    else:
      logits, model = payload, None
    seq_logits = tf.reshape(logits, (-1, seq_length, num_labels))
    transition_matrix = tf.compat.v1.get_variable(
      "transition_matrix", [num_labels, num_labels],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    sequence_lengths = tf.reduce_sum(label_weights, axis=-1)
    sequence_lengths = tf.cast(sequence_lengths, tf.int64)
    total_loss, _ = tf.contrib.crf.crf_log_likelihood(seq_logits, label_ids, sequence_lengths,
                                                      transition_matrix)
    total_loss = tf.reduce_mean(-total_loss)

    predictions, predicted_scores = tf.contrib.crf.crf_decode(seq_logits, transition_matrix, sequence_lengths)

    if FLAGS.do_adversarial_train and is_training:
      embedding_output = model.get_embedding_output()
      adv_loss = compute_adv_loss(embedding_output=embedding_output,
                                  labert_config=labert_config,
                                  input_ids=input_ids, input_mask=input_mask,
                                  start_positions=positional_embeddings_start,
                                  end_positions=positional_embeddings_end,
                                  num_labels=num_labels, label_positions=label_positions,
                                  label_weights=label_weights, is_training=is_training,
                                  target_logits=logits,
                                  noise_epsilon=1e-5, step_size=1e-3)
      total_loss = total_loss + adv_loss

    total_loss = tf.identity(total_loss, name='total_loss')

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
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

    return total_loss, predictions, predicted_scores, label_ids, label_weights

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    total_loss, predictions, predicted_scores, label_ids, label_weights = _model_fn(features, labels, mode, params)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
        loss=total_loss, init_lr=learning_rate,
        beta1=beta1, beta2=beta2, epsilon=epsilon, num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        hvd=None, use_fp16=False, num_accumulate_steps=1,
        optimizer_type="adam", allreduce_post_accumulation=False,
        lr_layer_decay_rate=lr_layer_decay_rate)

      logging_hook = LossLoggingHook(params['batch_size'], every_n_iter=int(num_train_steps / 200 + 1))

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=None,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, predictions, label_weights):
        label_weights = tf.reshape(label_weights, [-1])
        label_ids = tf.reshape(label_ids, [-1])
        predictions = tf.reshape(predictions, [-1])

        precision_micro = tf_metrics.precision(label_ids, predictions, num_labels,
                                               pos_indices=pos_indices, weights=label_weights, average="micro")
        recall_micro = tf_metrics.recall(label_ids, predictions, num_labels,
                                         pos_indices=pos_indices, weights=label_weights, average="micro")
        f_micro = tf_metrics.f1(label_ids, predictions, num_labels,
                                pos_indices=pos_indices, weights=label_weights, average="micro")

        accuracy = tf.compat.v1.metrics.accuracy(
          labels=label_ids, predictions=predictions, weights=label_weights)

        return {
          "eval_precision (micro)": precision_micro,
          "eval_recall (micro)": recall_micro,
          "eval_f (micro)": f_micro,
          "eval_acc": accuracy
        }

      eval_metrics = (metric_fn,
                      [total_loss, label_ids, predictions, label_weights])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=None)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": predicted_scores,
                       "predictions": predictions,
                       "label_weights": label_weights},
          scaffold_fn=None)
    return output_spec

  return model_fn


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  labert_config = modeling.BertConfig.from_json_file(FLAGS.labert_config_file)

  if FLAGS.max_seq_length > labert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, labert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  processor = SequenceLabelingProcessor()

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

  tf.compat.v1.logging.info("Number of labels: {}".format(len(label_list)))
  for label in label_list:
    tf.compat.v1.logging.info(" - {}".format(label))

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
    master=None,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    keep_checkpoint_max=1,
    log_step_count_steps=1 << 25,
    tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=8,
      per_host_input_for_training=is_per_host))

  pos_indices = [index for index, label in enumerate(label_list) if label.lower() != 'o']
  model_fn = model_fn_builder(
    labert_config=labert_config,
    num_labels=len(label_list),
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    beta1=FLAGS.adam_beta1,
    beta2=FLAGS.adam_beta2,
    epsilon=FLAGS.adam_epsilon,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    lr_layer_decay_rate=FLAGS.lr_layer_decay_rate,
    pos_indices=pos_indices)

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
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file,)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=False,
        num_training_instances=len(train_examples))

  eval_input_fn = None
  if FLAGS.do_eval and processor.has_dev_file(FLAGS.data_dir):
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
      eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file,)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(eval_examples), num_actual_eval_examples,
                              len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

  if FLAGS.do_train and FLAGS.do_eval and processor.has_dev_file(FLAGS.data_dir):
    best_ckpt_exporter = BestCheckpointCopier(name='best', checkpoints_to_keep=1,
                                              score_metric='eval_f (micro)',
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
    # This tells the estimator to run through the entire set.
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
    file_based_convert_examples_to_features(
      predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file)

    tf.compat.v1.logging.info("***** Running prediction *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(predict_examples), num_actual_predict_examples,
                              len(predict_examples) - num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.predict_output)
    with tf.io.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        predictions = prediction["predictions"]
        label_weights = prediction["label_weights"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\n".join(label_list[tag]
            for k, tag in enumerate(predictions) if label_weights[k] > 0.) + "\n\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("lexicon_file")
  flags.mark_flag_as_required("labert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run()

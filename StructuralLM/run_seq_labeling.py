# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=666
#SEED=123
from tfdeterminism import patch
patch()
import collections
import json
import math
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
from transformers import RobertaTokenizer
import modeling
import optimization
import numpy as np
np.random.seed(SEED)
from os.path import join as pjoin
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import tokenization
import six
import tensorflow as tf
tf.set_random_seed(SEED)
tf.random.set_random_seed(SEED)



flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string("labels", None,
                    "sequence labeling")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("roberta_model_path", None,
                    "Input raw text file (or comma-separated list of files).")
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("use_position_embeddings", False, "")

flags.DEFINE_bool("use_bbox_pos", True, "Whether to run training.")


flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")
flags.DEFINE_bool(
    "overwrite", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")
flags.DEFINE_bool("use_roberta", False, "Whether to use RoBERTa algorithm")


class InputExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               words,
               labels,
               boxes,
               pos_boxes,
               file_name):
    self.qas_id = qas_id
    self.words = words
    self.labels = labels
    self.boxes = boxes
    self.pos_boxes = pos_boxes
    self.file_name = file_name

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", words: %s" % (
        tokenization.printable_text(" ".join(self.words)))
    if labels:
      s += ", labels: %d" % (" ".join(self.labels))
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               example_index,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               pos_x0,
               pos_y0,
               pos_x1,
               pos_y1,
               pos_boxes,
               file_name):
    self.example_index = example_index
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.pos_x0 = pos_x0
    self.pos_y0 = pos_y0
    self.pos_x1 = pos_x1
    self.pos_y1 = pos_y1
    self.pos_bbox = pos_boxes
    self.file_name = file_name


def read_funsd_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  if is_training:
    mode = "train"
    #mode = "train_all"
  else:
    mode = "test"
    #mode = "test"
  #file_path = os.path.join(input_file, "{}.txt".format(mode))
  #box_file_path = os.path.join(input_file, "{}_box.txt".format(mode))
  file_path = input_file + ".txt"
  box_file_path = input_file + "_box.txt"
  guid_index = 1
  examples = []
  with open(file_path, encoding="utf-8") as f, open(
      box_file_path, encoding="utf-8"
  ) as fb:
    words = []
    labels = []
    boxes = []
    pos_boxes = []
    file_name = ""
    for line, bline in zip(f, fb):
      if line.startswith("-DOCSTART-") or line == "" or line == "\n":
        if words:
          if line.startswith("-DOCSTART-"):
            file_name = line.strip().split()[-1]
          examples.append(
              InputExample(
                   qas_id=guid_index,
                   words=words,
                   labels=labels,
                   boxes=boxes,
                   pos_boxes=pos_boxes,
                   file_name=file_name,
              )
          )
          guid_index += 1
          words = []
          boxes = []
          labels = []
          pos_boxes = []
      else:
        splits = line.split("\t")
        bsplits = bline.split("\t")
        assert len(splits) == 2
        assert len(bsplits) == 3
        assert splits[0] == bsplits[0]
        words.append(splits[0])
        if len(splits) > 1:
          labels.append(splits[-1].replace("\n", ""))
          box = bsplits[-2].replace("\n", "")
          box = [int(b) for b in box.split()]
          boxes.append(box)
          pos_box = int(bsplits[-1].strip("\n"))
          pos_boxes.append(pos_box)
        else:
          # Examples could have no label for mode = "test"
          labels.append("O")
    if words:
      examples.append(
        InputExample(
            qas_id=guid_index,
            words=words,
            labels=labels,
            boxes=boxes,
            pos_boxes=pos_boxes,
            file_name=file_name)
      )
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, label_list, cls_token_box=[0,0,0,0], 
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0], pad_token_label_id=-1):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {label: i for i, label in enumerate(label_list)}
  #pad_token_label_id = label_map[label_list[-1]]

  cnt_bad = 0
  for (example_index, example) in enumerate(examples):
    if example_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d", example_index, len(examples))
    token_boxes = []
    tokens = []
    label_ids = []
    token_pos_boxes = []
    cnt = 0
    for word, label, box, pos_box in zip(example.words, example.labels, example.boxes, example.pos_boxes):
      if FLAGS.use_roberta:
        word_tokens = tokenizer.tokenize("pad "+word)[1:]
      else:
        word_tokens = tokenizer.tokenize(word)
      tokens.extend(word_tokens)
      token_boxes.extend([box] * len(word_tokens))
      token_pos_boxes.extend([pos_box] * len(word_tokens))
      label_ids.extend(
        [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
      )
      cnt += 1
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
      tokens = tokens[: (max_seq_length - special_tokens_count)]
      token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
      token_pos_boxes = token_pos_boxes[: (max_seq_length - special_tokens_count)]
      label_ids = label_ids[: (max_seq_length - special_tokens_count)]
      cnt_bad += 1
      tf.logging.info("bad case %d" % cnt_bad)
      # end 
    if FLAGS.use_roberta:
      tokens += ["</s>"]
    else:
      tokens += ["[SEP]"]
    token_boxes += [sep_token_box]
    token_pos_boxes += [0]
    label_ids += [pad_token_label_id]
    segment_ids = [0] * len(tokens)
      # start
    if FLAGS.use_roberta:
      tokens = ["<s>"] + tokens
    else:
      tokens = ["[CLS]"] + tokens
    token_boxes = [cls_token_box] + token_boxes
    label_ids = [pad_token_label_id] + label_ids
    segment_ids = [0] + segment_ids
    token_pos_boxes = [0] + token_pos_boxes
        
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    pad_index = 1 if FLAGS.use_roberta else 0
    while len(input_ids) < max_seq_length:
      input_ids.append(pad_index)
      input_mask.append(pad_index)
      segment_ids.append(pad_index)
      label_ids.append(pad_token_label_id)
      token_boxes.append(pad_token_box)
      token_pos_boxes.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length
    assert len(token_pos_boxes) == max_seq_length
    pos_x0 = []
    pos_y0 = []
    pos_x1 = []
    pos_y1 = []
    for each in token_boxes:
        pos_x0.append(each[0])
        pos_y0.append(each[1])
        pos_x1.append(each[2])
        pos_y1.append(each[3])

    if example_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("example_index: %s" % (example_index))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info(
          "input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      tf.logging.info(
          "label_ids: %s" % " ".join([str(x) for x in label_ids]))
      tf.logging.info(
          "boxes: %s" % " ".join([str(x) for x in token_boxes]))
    feature = InputFeatures(
        example_index=example_index,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        pos_x0=pos_x0,
        pos_y0=pos_y0,
        pos_x1=pos_x1,
        pos_y1=pos_y1,
        pos_boxes=token_pos_boxes,
        file_name=example.file_name)

      # Run callback
    output_fn(feature)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings, pos_x0, pos_y0, pos_x1, pos_y1, pos_bbox, num_labels):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      pos_x0=pos_x0,
      pos_y0=pos_y0,
      pos_x1=pos_x1,
      pos_y1=pos_y1,
      pos_bbox=pos_bbox if FLAGS.use_bbox_pos else None,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32,
      use_roberta=FLAGS.use_roberta,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_position_embeddings=FLAGS.use_position_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02, seed=2222))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [num_labels], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, num_labels])

  return logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_labels):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    example_index = features["example_index"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    
    pos_x0 = features["pos_x0"]
    pos_y0 = features["pos_y0"]
    pos_x1 = features["pos_x1"]
    pos_y1 = features["pos_y1"]
    pos_bbox = features["pos_bbox"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    label_logits = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        pos_x0=pos_x0,
        pos_y0=pos_y0,
        pos_x1=pos_x1,
        pos_y1=pos_y1,
        pos_bbox=pos_bbox,
        num_labels=num_labels)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      label_ids = features["label_ids"]
      def compute_loss(logits, positions, num_labels):
        one_hot_positions = tf.one_hot(
            positions, depth=num_labels, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      total_loss = compute_loss(label_logits, label_ids, num_labels)

      #train_op = optimization.create_optimizer(
      #    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      label_ids = features["label_ids"]
      probabilities = tf.nn.softmax(label_logits, axis=-1)
      predicts = tf.argmax(probabilities, axis=-1)
      predictions = {
          "example_index": example_index,
          "label_logits": label_logits,
          "predicts": predicts,
          "label_ids": label_ids
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      "example_index":
          tf.FixedLenFeature([], tf.int64),
      "input_ids":
          tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask":
          tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([seq_length], tf.int64),
      "pos_x0":
          tf.FixedLenFeature([seq_length], tf.int64),
      "pos_y0":
          tf.FixedLenFeature([seq_length], tf.int64),
      "pos_x1":
          tf.FixedLenFeature([seq_length], tf.int64),
      "pos_y1":
          tf.FixedLenFeature([seq_length], tf.int64),
      "pos_bbox":
          tf.FixedLenFeature([seq_length], tf.int64)
    }

  #if is_training:
  name_to_features["label_ids"] = tf.FixedLenFeature([seq_length], tf.int64)

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
      #d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature
    features = collections.OrderedDict()
    features["example_index"] = create_int_feature([feature.example_index])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["pos_x0"] = create_int_feature(feature.pos_x0)
    features["pos_y0"] = create_int_feature(feature.pos_y0)
    features["pos_x1"] = create_int_feature(feature.pos_x1)
    features["pos_y1"] = create_int_feature(feature.pos_y1)
    features["pos_bbox"] = create_int_feature(feature.pos_bbox)

    features["label_ids"] = create_int_feature(feature.label_ids)
    #if self.is_training:
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

def get_labels(path):
  if path:
    with open(path, "r") as f:
      labels = f.read().splitlines()
    if "O" not in labels:
      labels = ["O"] + labels
    '''
    if "X" not in labels:
      labels += ["X"]
    '''
    return labels
  else:
    return [
        "O",
        "B-MISC",
        "I-MISC",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
    ]

def main(_):
  
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)
  if FLAGS.overwrite and FLAGS.do_train:
    os.system("rm -rf %s" % pjoin(FLAGS.output_dir, "model.*"))
    os.system("rm -rf %s" % pjoin(FLAGS.output_dir, "graph*"))
    os.system("rm -rf %s" % pjoin(FLAGS.output_dir, "events*"))
    os.system("rm -rf %s" % pjoin(FLAGS.output_dir, "checkp*"))
  

  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.use_roberta: 
    tokenizer = RobertaTokenizer.from_pretrained(FLAGS.roberta_model_path, do_lower_case=False)
  else:
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # Prepare CONLL-2003 task
  labels = get_labels(FLAGS.labels)
  num_labels = len(labels)
  if FLAGS.do_train:
    train_examples = read_funsd_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    '''
    rng = random.Random(12345)
    rng.shuffle(train_examples)
    '''

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      num_labels=num_labels)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  pad_token_label_id = -1
  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    #pad_token_label_id = CrossEntropyLoss().ignore_index


    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature,
        label_list=labels)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    eval_examples = read_funsd_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature,
        label_list=labels)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    preds = None
    out_label_ids = None
    file_name_list = []
    all_tokens_list = []
    all_tokens_map_list = []
    for i, result in enumerate(estimator.predict(
        predict_input_fn, yield_single_examples=True)):
      eval_feature = eval_features[i]
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      example_index = int(result["example_index"])
      file_name_list.append(eval_feature.file_name)
      # [1, sequence_lenth, num_labels]
      label_logits = np.expand_dims(result["label_logits"], axis=0)
      #[1, sequence_lenth]
      label_ids = np.expand_dims(result["label_ids"], axis=0)
      if preds is None:
        preds = label_logits
        out_label_ids = label_ids
      else:
        preds = np.append(preds, label_logits, axis=0)
        out_label_ids = np.append(out_label_ids, label_ids, axis=0)
    # calculate score
    preds = np.argmax(preds, axis=2)
    
    label_map = {i: label for i, label in enumerate(labels)}
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
          out_label_list[i].append(label_map[out_label_ids[i][j]])
          preds_list[i].append(label_map[preds[i][j]])
    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    
    tf.logging.info("***** Eval results  Epoch %d*****", int(FLAGS.num_train_epochs))
    for key in sorted(results.keys()):
        tf.logging.info("  %s = %s", key, str(results[key]))
    
    
if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

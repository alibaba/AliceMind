# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors.
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
import tensorflow as tf
import collections
import tokenization


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def write_lattice_instance_to_example_file(
    instance, tokenizer, writer, max_seq_length,
    max_predictions_per_seq,
    position_embedding_names=('start', 'end'), do_dump_example=False):
  """Create TF example files from `TrainingInstance`s."""

  input_ids = tokenizer.convert_tokens_to_ids(instance.encodings.tokens)
  positional_embeddings = instance.encodings.position_embedding(position_embedding_names)
  for positional_embedding in positional_embeddings:
    for i in positional_embedding:
      assert i >= 0, f"{instance.encodings.tokens}"
  input_mask = [1] * len(input_ids)
  segment_ids = list(instance.segment_ids)
  assert len(input_ids) <= max_seq_length

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    for t in positional_embeddings:
      t.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  for positional_embedding in positional_embeddings:
    assert len(positional_embedding) == max_seq_length

  masked_lm_positions = list(instance.masked_lm_positions)
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
  masked_lm_weights = [1.0] * len(masked_lm_ids)

  while len(masked_lm_positions) < max_predictions_per_seq:
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  next_sentence_label = instance.next_sentence_label

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  for name, positional_embedding in zip(position_embedding_names, positional_embeddings):
    features[f"positional_embeddings_{name}"] = create_int_feature(positional_embedding)
  features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
  features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
  features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
  features["next_sentence_labels"] = create_int_feature([next_sentence_label])

  assert all([len(t) == len(input_ids) for t in positional_embeddings])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))

  writer.write(tf_example.SerializeToString())

  if do_dump_example:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in instance.encodings.tokens]))

    for feature_name in features.keys():
      feature = features[feature_name]
      values = []
      if feature.int64_list.value:
        values = feature.int64_list.value
      elif feature.float_list.value:
        values = feature.float_list.value
      tf.compat.v1.logging.info(
        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))


def write_lattice_instances_to_example_files(
    instances, tokenizer, max_seq_length,
    max_predictions_per_seq, output_files,
    position_embedding_names=('start', 'end')):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    write_lattice_instance_to_example_file(
      instance, tokenizer, writers[writer_index],
      max_seq_length, max_predictions_per_seq,
      position_embedding_names, inst_index < 20)
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

  for writer in writers:
    writer.close()

  tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def write_instances_to_example_files(instances, tokenizer, max_seq_length,
                                     max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.encodings)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = instance.next_sentence_label

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 400:
      tf.compat.v1.logging.info("*** Example ***")
      tf.compat.v1.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.encodings]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.compat.v1.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.compat.v1.logging.info("Wrote %d total instances", total_written)


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, encodings, segment_ids, masked_lm_positions, masked_lm_labels,
               next_sentence_label):
    self.encodings = encodings
    self.segment_ids = segment_ids
    self.next_sentence_label = next_sentence_label
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
      [tokenization.printable_text(x) for x in self.encodings.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "next_sentence_label: %s\n" % self.next_sentence_label
    s += "masked_lm_positions: %s\n" % (" ".join(
      [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
      [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

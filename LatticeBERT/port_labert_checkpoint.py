# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors. All rights reserved.
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
import argparse
import tensorflow as tf
import modeling
import modeling_labert
import tokenization_labert
import collections
import re
import os
import json
import shutil


def main():
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  cmd = argparse.ArgumentParser()
  cmd.add_argument('--init_checkpoint', required=True, help="")
  cmd.add_argument('--lexicon', required=True, help="the path to the target domain lexicon.")
  cmd.add_argument('--output_dir', required=True, help="")
  args = cmd.parse_args()

  input_ids = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="input_ids")
  input_mask = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="input_mask")
  segment_ids = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="segment_ids")
  positional_embeddings_start = tf.compat.v1.placeholder(
    dtype=tf.int64, shape=[None, None], name="positional_embeddings_start")
  positional_embeddings_end = tf.compat.v1.placeholder(
    dtype=tf.int64, shape=[None, None], name="positional_embeddings_end")

  os.makedirs(args.output_dir, exist_ok=True)

  tf.compat.v1.logging.info("Creating source tokenizer ...")
  src_tokenizer = tokenization_labert.LatticeTokenizerWithMapping(
    vocab_file=os.path.join(args.init_checkpoint, "vocab.txt"),
    lexicon_file=os.path.join(args.init_checkpoint, "lexicon.txt"),
    do_lower_case=True)

  tf.compat.v1.logging.info("Creating target tokenizer ...")
  tgt_tokenizer = tokenization_labert.LatticeTokenizerWithMapping(
    vocab_file=os.path.join(args.init_checkpoint, "vocab.txt"),
    lexicon_file=args.lexicon,
    do_lower_case=True)

  # Copy into lexicon
  shutil.copy(args.lexicon,
              os.path.join(args.output_dir, "lexicon.txt"))

  shutil.copy(os.path.join(args.init_checkpoint, "vocab.txt"),
              os.path.join(args.output_dir, "vocab.txt"))

  with open(os.path.join(args.init_checkpoint, "labert_config.json"), 'r') as reader:
    labert_config = json.load(reader)

  labert_config["vocab_size"] = len(tgt_tokenizer.vocab)

  # Reset the configuration.
  with open(os.path.join(args.output_dir, "labert_config.json"), 'w') as writer:
    json.dump(labert_config, writer)

  source_labert_config = modeling_labert.LaBertConfig.from_json_file(
    os.path.join(args.init_checkpoint, "labert_config.json"))
  target_labert_config = modeling_labert.LaBertConfig.from_json_file(
    os.path.join(args.output_dir, "labert_config.json"))

  model = modeling_labert.LaBertModel(
    config=target_labert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    start_positions=positional_embeddings_start,
    end_positions=positional_embeddings_end,
    use_one_hot_embeddings=False, )

  output_layer = model.get_pooled_output()

  with tf.compat.v1.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform"):
      input_tensor = tf.layers.dense(
        output_layer,
        units=target_labert_config.embedding_size,
        activation=modeling.get_activation(target_labert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
          target_labert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.compat.v1.get_variable(
      "output_bias",
      shape=[target_labert_config.vocab_size],
      initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("cls/seq_relationship"):
    tf.compat.v1.get_variable(
      "output_weights",
      shape=[3, target_labert_config.hidden_size],
      initializer=modeling.create_initializer(target_labert_config.initializer_range))
    tf.compat.v1.get_variable(
      "output_bias", shape=[3], initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("bert-1/embeddings"):
    word_embeddings = tf.compat.v1.get_variable(
      "word_embeddings",
      shape=[source_labert_config.vocab_size, source_labert_config.embedding_size],
      initializer=modeling.create_initializer(target_labert_config.initializer_range)
    )

  with tf.compat.v1.variable_scope("cls-1/predictions"):
    output_bias1 = tf.compat.v1.get_variable(
      "output_bias", shape=[source_labert_config.vocab_size],
      initializer=tf.zeros_initializer())

  target_tvars = tf.compat.v1.trainable_variables()
  name_to_variable = collections.OrderedDict()
  for var in target_tvars:
    name = var.name
    if name in ('cls/predictions/output_bias',
                'bert/embeddings/word_embeddings'):
      continue
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  src_ckpt_vars = tf.train.list_variables(args.init_checkpoint)
  assignment_map = collections.OrderedDict()

  for src_ckpt_var in src_ckpt_vars:
    name = src_ckpt_var[0]
    if name in ('cls/predictions/output_bias',
                'bert/embeddings/word_embeddings'):
      fields = name.split('/')
      fields[0] = fields[0] + '-1'
      new_name = "/".join(fields)
    else:
      new_name = name

    if new_name not in name_to_variable:
      continue

    assignment_map[name] = name_to_variable[new_name]

  for name in assignment_map:
    tf.compat.v1.logging.info(f'{name:60s}{assignment_map[name]}')

  tf.compat.v1.train.init_from_checkpoint(args.init_checkpoint, assignment_map)

  index_mapping = collections.OrderedDict()
  source_indices = []
  target_indices = []
  for token in src_tokenizer.vocab:
    if token in tgt_tokenizer.vocab:
      source_index = src_tokenizer.convert_tokens_to_ids([token])[0]
      target_index = tgt_tokenizer.convert_tokens_to_ids([token])[0]
      index_mapping[source_index] = target_index

      source_indices.append(source_index)
      target_indices.append(target_index)

  tf.compat.v1.logging.info(f'{len(index_mapping)} matched.')
  with open(os.path.join(args.output_dir, 'lexicon_mapping.json'), 'w') as writer:
    json.dump(index_mapping, writer)

  update_op1 = tf.compat.v1.scatter_update(model.embedding_table,
                                           target_indices,
                                           tf.compat.v1.gather(word_embeddings, source_indices, axis=0))

  update_op2 = tf.compat.v1.scatter_update(output_bias,
                                           target_indices,
                                           tf.compat.v1.gather(output_bias1, source_indices, axis=0))

  with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(update_op1)
    session.run(update_op2)

    saver = tf.compat.v1.train.Saver()
    saver.save(session, os.path.join(args.output_dir, "model.ckpt"))


if __name__ == "__main__":
  main()

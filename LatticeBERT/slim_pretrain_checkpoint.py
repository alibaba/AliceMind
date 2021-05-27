# coding=utf-8
# Copyright 2021 The ALICE Alibaba DAMO NLP Team Authors.
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
import modeling
import modeling_labert
import os
import shutil
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("init_checkpoint", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_string("embedding_postprocess_type", "bert", "The postprocess of embedding.")


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  input_ids = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="input_ids")
  input_mask = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="input_mask")
  segment_ids = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, None], name="segment_ids")
  positional_embeddings_start = tf.compat.v1.placeholder(
    dtype=tf.int64, shape=[None, None], name="positional_embeddings_start")
  positional_embeddings_end = tf.compat.v1.placeholder(
    dtype=tf.int64, shape=[None, None], name="positional_embeddings_end")

  labert_config = modeling_labert.LaBertConfig.from_json_file(os.path.join(FLAGS.init_checkpoint, 'labert_config.json'))
  model = modeling_labert.LaBertModel(
    config=labert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    start_positions=positional_embeddings_start,
    end_positions=positional_embeddings_end,
    use_one_hot_embeddings=False,)

  output_layer = model.get_pooled_output()

  with tf.compat.v1.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform"):
      input_tensor = tf.layers.dense(
        output_layer,
        units=labert_config.embedding_size,
        activation=modeling.get_activation(labert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
          labert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.compat.v1.get_variable(
      "output_bias",
      shape=[labert_config.vocab_size],
      initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("cls/seq_relationship"):
    output_weights = tf.compat.v1.get_variable(
      "output_weights",
      shape=[3, labert_config.hidden_size],
      initializer=modeling.create_initializer(labert_config.initializer_range))
    output_bias = tf.compat.v1.get_variable(
      "output_bias", shape=[3], initializer=tf.zeros_initializer())

  tvars = tf.compat.v1.trainable_variables()
  (assignment_map, initialized_variable_names
   ) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)

  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

  tf.compat.v1.logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

  with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver()
    dirname = os.path.dirname(FLAGS.output_file)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    saver.save(session, FLAGS.output_file)
    for metafile in ['labert_config.json', 'vocab.txt', 'lexicon.txt']:
      shutil.copy(os.path.join(FLAGS.init_checkpoint, metafile),
                  os.path.join(dirname, metafile))


if __name__ == "__main__":
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.compat.v1.app.run()


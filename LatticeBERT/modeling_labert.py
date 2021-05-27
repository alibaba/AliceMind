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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import tensorflow as tf
from modeling import (BertConfig,
                      reshape_to_matrix,
                      reshape_from_matrix,
                      get_shape_list,
                      get_activation,
                      gelu,
                      dropout,
                      layer_norm,
                      layer_norm_and_dropout,
                      create_attention_mask_from_input_mask,
                      create_initializer)
from gpu_environment import get_custom_getter


class LaBertConfig(BertConfig):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               embedding_size=128,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    super().__init__(vocab_size=vocab_size,
                     hidden_size=hidden_size,
                     num_hidden_layers=num_hidden_layers,
                     num_attention_heads=num_attention_heads,
                     intermediate_size=intermediate_size,
                     hidden_act=hidden_act,
                     hidden_dropout_prob=hidden_dropout_prob,
                     attention_probs_dropout_prob=attention_probs_dropout_prob,
                     max_position_embeddings=max_position_embeddings,
                     type_vocab_size=type_vocab_size,
                     initializer_range=initializer_range)
    self.embedding_size = embedding_size


class LaBertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling_labert.LaBertConfig(vocab_size=32000, hidden_size=512,
    embedding_size=128, num_hidden_layers=8, num_attention_heads=6,
    intermediate_size=1024)

  model = modeling_labert.LaBertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config: LaBertConfig,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               start_positions=None,
               end_positions=None,
               use_one_hot_embeddings=False,
               scope=None,
               compute_type=tf.float32,
               transformer_model_type="post-ln",
               do_share_parameter_across_layers=False,
               do_return_all_attention_maps=False, ):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".
      compute_type: (optional) either float32 or float16. Only applies to GPUs.
      do_return_all_attention_maps:

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    self.do_return_all_attention_maps = do_return_all_attention_maps

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.compat.v1.variable_scope(scope, default_name="bert",
                                     custom_getter=get_custom_getter(compute_type)):
      with tf.compat.v1.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = factorized_embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            embedding_size=config.embedding_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
          input_tensor=self.embedding_output,
          use_token_type=True,
          token_type_ids=token_type_ids,
          token_type_vocab_size=config.type_vocab_size,
          token_type_embedding_name="token_type_embeddings",
          positions=[start_positions, end_positions],
          use_position_embeddings=False,  # set to false due to the usage of relative position embeddings.
          position_embedding_name=["position_embeddings_start", "position_embeddings_end"],
          initializer_range=config.initializer_range,
          max_position_embeddings=config.max_position_embeddings,
          dropout_prob=config.hidden_dropout_prob)

      with tf.compat.v1.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
          input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        transformer_model_fn = transformer_model if transformer_model_type == "post-ln" else transformer_model_preln

        payload = transformer_model_fn(
          compute_type=compute_type,
          position_embeddings_ids=[start_positions, end_positions],
          max_position_embeddings=config.max_position_embeddings,
          embedding_size=config.embedding_size,
          input_tensor=tf.saturate_cast(self.embedding_output, compute_type),
          attention_mask=attention_mask,
          hidden_size=config.hidden_size,
          num_hidden_layers=config.num_hidden_layers,
          num_attention_heads=config.num_attention_heads,
          intermediate_size=config.intermediate_size,
          intermediate_act_fn=get_activation(config.hidden_act),
          hidden_dropout_prob=config.hidden_dropout_prob,
          attention_probs_dropout_prob=config.attention_probs_dropout_prob,
          initializer_range=config.initializer_range,
          do_share_parameter_across_layers=do_share_parameter_across_layers,
          do_return_all_layers=True,
          do_return_attention_maps=do_return_all_attention_maps,)
        if do_return_all_attention_maps:
          self.all_encoder_layers, self.all_encoder_attention_maps = payload
        else:
          self.all_encoder_layers = payload

      self.sequence_output = tf.cast(self.all_encoder_layers[-1], tf.float32)
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.compat.v1.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.compat.v1.layers.dense(
          first_token_tensor,
          config.hidden_size,
          activation=tf.tanh,
          kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_all_encoder_attention_maps(self):
    if not self.do_return_all_attention_maps:
      raise ValueError("return attention maps was not configure when building graph")
    return self.all_encoder_attention_maps

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def factorized_embedding_lookup(input_ids,
                                vocab_size,
                                hidden_size=128,
                                embedding_size=128,
                                initializer_range=0.02,
                                word_embedding_name="word_embeddings",
                                use_one_hot_embeddings=False):
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  factorized_embedding_table = tf.compat.v1.get_variable(
    name=word_embedding_name,
    shape=[vocab_size, embedding_size],
    initializer=create_initializer(initializer_range))

  projection = tf.compat.v1.get_variable(
    name=word_embedding_name + "_projection",
    shape=[embedding_size, hidden_size],
    initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, factorized_embedding_table)
  else:
    output = tf.gather(factorized_embedding_table, flat_input_ids)

  output = tf.matmul(output, projection)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * hidden_size])
  return (output, factorized_embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            positions=None,
                            use_position_embeddings=True,
                            position_embedding_name=None,
                            initializer_range=0.02,
                            max_position_embeddings=1024,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.compat.v1.get_variable(
      name=token_type_embedding_name,
      shape=[token_type_vocab_size, width],
      initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    if type(positions) in [list, tuple]:
      assert type(position_embedding_name) in [list, tuple]
    else:
      assert type(position_embedding_name) is str
      positions = [positions]
      position_embedding_name = [position_embedding_name]
    if len(positions) == 1:
      widths = [width]
    else:
      widths = [width // len(positions)] * len(positions)
      widths[-1] = width - sum(widths[:-1])

    assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = [tf.compat.v1.get_variable(
        name=position_embedding_name[i],
        shape=[max_position_embeddings, widths[i]],
        initializer=create_initializer(initializer_range)) for i in range(len(positions))]

      all_positional_embeddings = []
      for i in range(len(positions)):
        position_embeddings_id = positions[i]
        if position_embeddings_id.shape.ndims == 2:
          position_embeddings_id = tf.expand_dims(position_embeddings_id, axis=[-1])
        # print(position_embeddings_id.shape) # (8, 128, 1)

        flat_input_ids = tf.reshape(position_embeddings_id, [-1])
        # print(flat_input_ids.shape) # (1024,) = 8*128

        position_embeddings = tf.gather(full_position_embeddings[i], flat_input_ids)
        # (1024, 768)

        input_shape = get_shape_list(position_embeddings_id)

        position_embeddings = tf.reshape(position_embeddings,
                                         input_shape[0:-1] + [input_shape[-1] * widths[i]])
        all_positional_embeddings.append(position_embeddings)
        # (8, 128, 384)
      position_embeddings = tf.concat(all_positional_embeddings, -1)
      # print(position_embeddings.shape) # (8, 128, 768)

      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def attention_layer_with_reset_attention_scores(from_tensor,
                                                to_tensor,
                                                reset_attention_scores,
                                                compute_type=tf.float32,
                                                attention_mask=None,
                                                num_attention_heads=1,
                                                size_per_head=512,
                                                query_act=None,
                                                key_act=None,
                                                value_act=None,
                                                attention_probs_dropout_prob=0.0,
                                                initializer_range=0.02,
                                                do_return_2d_tensor=False,
                                                batch_size=None,
                                                from_seq_length=None,
                                                to_seq_length=None,
                                                do_return_attention_maps=False, ):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    do_return_attention_maps: Whether to also return attention map of all layers.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
      input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
      "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
        "When passing in rank 2 tensors to attention_layer, the values "
        "for `batch_size`, `from_seq_length`, and `to_seq_length` "
        "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.compat.v1.layers.dense(
    from_tensor_2d,
    num_attention_heads * size_per_head,
    activation=query_act,
    name="query",
    kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=key_act,
    name="key",
    kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=value_act,
    name="value",
    kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))
  # print(attention_scores.shape) # (2, 4, 128, 128)
  # positional_attention_score
  # compute_type
  attention_scores = attention_scores + reset_attention_scores

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)
  attention_probs = tf.saturate_cast(attention_probs, compute_type)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  if do_return_attention_maps:
    return context_layer, attention_probs
  return context_layer


def compute_reset_attention_scores(position_embeddings_ids, max_position_embeddings, embedding_size,
                                   num_attention_heads, size_per_head, initializer_range,
                                   max_relative_position=128):
  # batch_size: B
  # embedding_size: E
  # seq_length: L
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
      input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  if position_embeddings_ids is not None:
    assert type(position_embeddings_ids) in (list, tuple)
    assert len(position_embeddings_ids) == 2

  start_position_embedding_table = tf.compat.v1.get_variable(
    name='start_position_embeddings',
    shape=[max_position_embeddings, embedding_size],
    initializer=create_initializer(initializer_range))
  end_position_embedding_table = tf.compat.v1.get_variable(
    name='end_position_embeddings',
    shape=[max_position_embeddings, embedding_size],
    initializer=create_initializer(initializer_range))

  position_embedding_tables = [start_position_embedding_table, end_position_embedding_table]

  position_embedding_outputs = []
  for i in range(len(position_embeddings_ids)):
    # position_embeddings_id: [B, L]
    position_embeddings_id = position_embeddings_ids[i]
    if position_embeddings_id.shape.ndims == 2:
      position_embeddings_id = tf.expand_dims(position_embeddings_id, axis=[-1])

    # flat_input_ids: [B*L]
    flat_input_ids = tf.reshape(position_embeddings_id, [-1])

    # position_embeddings: [B*L, E]
    position_embeddings = tf.gather(position_embedding_tables[i], flat_input_ids)

    # input_shape= (B, L, 1)
    input_shape = get_shape_list(position_embeddings_id)

    # position_embeddings: [B,L,E]
    position_embeddings = tf.reshape(position_embeddings,
                                     input_shape[0:-1] + [input_shape[-1] * embedding_size])

    position_embedding_outputs.append(position_embeddings)

  # position_embeddings: [B,L,2*E]
  position_embeddings = tf.concat(position_embedding_outputs, axis=-1)

  from_tensor = to_tensor = position_embeddings
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
      "The rank of `from_tensor` must match the rank of `to_tensor`.")

  assert len(from_shape) == 3

  batch_size = from_shape[0]
  from_seq_length = from_shape[1]
  to_seq_length = to_shape[1]

  # `from_tensor_2d` = [B*L,2*E]
  from_tensor_2d = reshape_to_matrix(position_embeddings)
  # `to_tensor_2d` = [B*L,2*E]
  to_tensor_2d = reshape_to_matrix(position_embeddings)

  # `query_layer` = [B*L, N*H]
  query_layer = tf.compat.v1.layers.dense(
    from_tensor_2d,
    num_attention_heads * size_per_head,
    activation=None,
    name="query_position",
    kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*L, N*H]
  key_layer = tf.compat.v1.layers.dense(
    to_tensor_2d,
    num_attention_heads * size_per_head,
    activation=None,
    name="key_position",
    kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, L, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, L, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, L, L]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  def build_relative_positions(range_vec_k, range_vec_q, max_relative_position):
    # """Generates matrix of relative positions between inputs."""
    # print(range_vec_k.shape, range_vec_q.shape)     # (2, 128) (2, 128)
    distance_mat = range_vec_k[:, None, :] - range_vec_q[:, :, None]
    # print('distance_mat', distance_mat.shape)       # distance_mat (2, 128, 128)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat

  start_positions, end_positions = position_embeddings_ids
  pairs = [[start_positions, start_positions],
           [start_positions, end_positions],
           [end_positions, start_positions],
           [end_positions, end_positions]]

  relative_position_scores = []

  # Generates scores for each relative position of dimension depth.
  for i, (from_position, to_position) in enumerate(pairs):
    relative_position_score_table = tf.compat.v1.get_variable(
      name=f"relative_position_scores_{i}",
      shape=[max_relative_position * 2 + 1, num_attention_heads],
      initializer=create_initializer(initializer_range))

    # `relative_positions_matrix` = [B, L, L]
    relative_positions_matrix = build_relative_positions(
      from_position, to_position,
      max_relative_position=max_relative_position)

    # `relative_position_score` = [B, L, L, N]
    relative_position_score = tf.gather(relative_position_score_table, relative_positions_matrix)
    # `relative_position_score` = [B, N, L, L]
    relative_position_score = tf.transpose(relative_position_score, [0, 3, 1, 2])

    relative_position_scores.append(relative_position_score)

  from_position_head = start_positions[:, None, :]  # (B, 1, L)
  from_position_tail = end_positions[:, None, :]    # (B, 1, L)
  to_position_head = start_positions[:, :, None]    # (B, L, 1)
  to_position_tail = end_positions[:, :, None]      # (B, L, 1)

  # `relative_position_encoding` = (B, L, L)
  relative_position_encoding = \
    1 * tf.cast(tf.greater_equal(from_position_head, to_position_head), tf.int64) + \
    2 * tf.cast(tf.greater(from_position_head, to_position_tail), tf.int64) + \
    4 * tf.cast(tf.greater_equal(from_position_tail, to_position_head), tf.int64) + \
    8 * tf.cast(tf.greater_equal(from_position_tail, to_position_tail), tf.int64) + \
    16 * (tf.cast(tf.equal(from_position_head, to_position_head), tf.int64) *
          tf.cast(tf.equal(from_position_tail, to_position_tail), tf.int64))

  # use 2^5 to encode relative position codes
  # 32 is the magic number
  relative_position_encoding_embedding_table = tf.compat.v1.get_variable(
    name="relative_type_embeddings_d1",
    shape=[32, num_attention_heads],
    initializer=create_initializer(initializer_range))

  # `relative_position_encoding_score` = (B, L, L, N)
  relative_position_encoding_score = tf.gather(relative_position_encoding_embedding_table,
                                               relative_position_encoding)

  # `relative_position_encoding_score` = (B, N, L, L)
  relative_position_encoding_score = tf.transpose(relative_position_encoding_score, [0, 3, 1, 2])

  # `unreset_attention_scores` = (B, N, L, L)
  unreset_attention_scores = relative_position_encoding_score + sum(relative_position_scores) + attention_scores

  in_reset_shape = tf.shape(unreset_attention_scores)

  reset_theta_1_raw = tf.compat.v1.get_variable(
    name="relative_att_reset_theta_1",
    shape=[1, 1, 1, 2 * embedding_size],
    initializer=create_initializer(initializer_range))
  reset_theta_2_raw = tf.compat.v1.get_variable(
    name="relative_att_reset_theta_2",
    shape=[1, 1, 1, 2 * embedding_size],
    initializer=create_initializer(initializer_range))

  def f21(input_tensor):
    # `q_l` = [1*1, N*H]
    q_l = tf.layers.dense(
      input_tensor,
      num_attention_heads * size_per_head,
      activation=None,
      name="query_position",
      kernel_initializer=create_initializer(initializer_range), reuse=True)

    # `key_layer` = [B*T, N*H]
    k_l = tf.layers.dense(
      input_tensor,
      num_attention_heads * size_per_head,
      activation=None,
      name="key_position",
      kernel_initializer=create_initializer(initializer_range), reuse=True)

    q_l = transpose_for_scores(q_l, 1, num_attention_heads, 1, size_per_head)
    k_l = transpose_for_scores(k_l, 1, num_attention_heads, 1, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(q_l, k_l, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))
    return attention_scores

  reset_theta_1 = f21(reset_theta_1_raw)
  reset_theta_2 = f21(reset_theta_2_raw)

  reset_theta_1 = tf.tile(reset_theta_1, [in_reset_shape[0], 1, 1, in_reset_shape[-1] - 1])
  reset_theta_2 = tf.tile(reset_theta_2, [in_reset_shape[0], 1, in_reset_shape[-1], 1])

  unreset_attention_scores = unreset_attention_scores[:, :, 1:, 1:]
  # print(in_reset.shape) # (2, 4, 127, 127)
  reset_attention_scores = tf.concat([reset_theta_1, unreset_attention_scores], axis=-2)
  # print(in_reset.shape) # (2, 4, 128, 127)
  reset_attention_scores = tf.concat([reset_theta_2, reset_attention_scores], axis=-1)
  # print(in_reset.shape) # (2, 4, 128, 128)

  return reset_attention_scores  # , key_layer_rel


def transformer_model(input_tensor,
                      compute_type,
                      position_embeddings_ids=None,
                      attention_mask=None,
                      max_position_embeddings=None,
                      embedding_size=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_share_parameter_across_layers=False,
                      do_return_all_layers=False,
                      do_return_attention_maps=False,):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_share_parameter_across_layers:
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    do_return_attention_maps: Whether to also return attention map of all layers.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  reset_attention_scores = compute_reset_attention_scores(position_embeddings_ids,
                                                          max_position_embeddings, embedding_size,
                                                          num_attention_heads=num_attention_heads,
                                                          size_per_head=attention_head_size,
                                                          initializer_range=initializer_range)

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  all_layer_attention_maps = []
  for layer_idx in range(num_hidden_layers):
    if do_share_parameter_across_layers:
      name_variable_scope = "layer_shared"
    else:
      name_variable_scope = "layer_%d" % layer_idx

    with tf.compat.v1.variable_scope(name_variable_scope,
                                     reuse=do_share_parameter_across_layers and layer_idx > 0):
      layer_input = prev_output

      with tf.compat.v1.variable_scope("attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
          payload = attention_layer_with_reset_attention_scores(
            from_tensor=layer_input,
            to_tensor=layer_input,
            reset_attention_scores=tf.saturate_cast(reset_attention_scores, compute_type),
            compute_type=compute_type,
            attention_mask=attention_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            batch_size=batch_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            do_return_attention_maps=do_return_attention_maps,)

          if do_return_attention_maps:
            attention_head, attention_prob = payload
            attention_heads.append(attention_head)
            all_layer_attention_maps.append(attention_prob)
          else:
            attention_head = payload
            attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.compat.v1.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.compat.v1.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.compat.v1.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    if do_return_attention_maps:
      return final_outputs, all_layer_attention_maps
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def transformer_model_preln(input_tensor,
                            compute_type,
                            position_embeddings_ids=None,
                            attention_mask=None,
                            max_position_embeddings=None,
                            hidden_size=768,
                            embedding_size=128,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            intermediate_act_fn=gelu,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            initializer_range=0.02,
                            do_share_parameter_across_layers=False,
                            do_return_all_layers=False,
                            do_return_attention_maps=False,):
  """Transformer model from "On layer normalization in the transformer architecture".

  See the original paper:
  https://openreview.net/pdf?id=B1x8anVFPr

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_share_parameter_across_layers:
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    do_return_attention_maps: Whether to also return attention map of all layers.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  reset_attention_scores = compute_reset_attention_scores(position_embeddings_ids,
                                                          max_position_embeddings, embedding_size,
                                                          num_attention_heads=num_attention_heads,
                                                          size_per_head=attention_head_size,
                                                          initializer_range=initializer_range)

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  all_layer_attention_maps = []
  for layer_idx in range(num_hidden_layers):
    if do_share_parameter_across_layers:
      name_variable_scope = "layer_shared"
    else:
      name_variable_scope = "layer_%d" % layer_idx

    with tf.compat.v1.variable_scope(name_variable_scope,
                                     reuse=do_share_parameter_across_layers and layer_idx > 0):
      layer_input = prev_output

      with tf.compat.v1.variable_scope("attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
          layer_input = layer_norm(layer_input)

          payload = attention_layer_with_reset_attention_scores(
              from_tensor=layer_input,
              to_tensor=layer_input,
              reset_attention_scores=tf.saturate_cast(reset_attention_scores, compute_type),
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              do_return_attention_maps=do_return_attention_maps,)
          if do_return_attention_maps:
            attention_head, attention_prob = payload
            attention_heads.append(attention_head)
            all_layer_attention_maps.append(attention_prob)
          else:
            attention_head = payload
            attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.compat.v1.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = attention_output + layer_input

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.compat.v1.variable_scope("intermediate"):
        attention_output = layer_norm(attention_output)
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.compat.v1.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_output + attention_output
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    if do_return_attention_maps:
      return final_outputs, all_layer_attention_maps
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

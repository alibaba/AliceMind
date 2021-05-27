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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops


def create_optimizer(loss, init_lr, beta1, beta2, epsilon,
                     num_train_steps, num_warmup_steps,
                     hvd, use_fp16, num_accumulate_steps=1,
                     optimizer_type="adam", allreduce_post_accumulation=False,
                     lr_layer_decay_rate=1.0,
                     ignore_pooler=False):
  """Creates an optimizer training op."""
  global_step = tf.compat.v1.train.get_or_create_global_step()

  if optimizer_type == "adam":
    power = 1.0
    decayed_learning_rate_at_crossover_point = init_lr * (
      (1.0 - float(num_warmup_steps) / float(num_train_steps)) ** power)
  else:
    power = 0.5
    decayed_learning_rate_at_crossover_point = init_lr

  adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
  tf.compat.v1.logging.info('decayed_learning_rate_at_crossover_point = {:e}, adjusted_init_lr = {:e}'.format(
    decayed_learning_rate_at_crossover_point, adjusted_init_lr))
  learning_rate = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=power,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  tf.compat.v1.logging.info("Initializing ADAM weight decay optimizer (v2)")
  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizerV2(
    learning_rate=learning_rate,
    weight_decay_rate=0.01,
    beta_1=beta1,
    beta_2=beta2,
    epsilon=epsilon,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    manual_fp16=use_fp16)

  if hvd is not None and (num_accumulate_steps == 1 or (not allreduce_post_accumulation)):
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         sparse_as_dense=True,
                                         compression=hvd.Compression.fp16 if use_fp16 else hvd.Compression.none)
  if use_fp16:
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
      init_loss_scale=2 ** 32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

  tvars = tf.compat.v1.trainable_variables()
  if ignore_pooler:
    tvars = [tvar for tvar in tvars if not tvar.name.startswith('bert/pooler/')]
  grads_and_vars = optimizer.compute_gradients(loss * 1.0 / num_accumulate_steps, tvars)

  if num_accumulate_steps > 1:
    local_step = tf.compat.v1.get_variable(
      name="local_step", shape=[], dtype=tf.int32, trainable=False, initializer=tf.zeros_initializer())
    batch_finite = tf.compat.v1.get_variable(
      name="batch_finite", shape=[], dtype=tf.bool, trainable=False, initializer=tf.ones_initializer())
    accum_vars = [tf.compat.v1.get_variable(
      name=tvar.name.split(":")[0] + "/accum",
      shape=tvar.shape.as_list(),
      dtype=tf.float32,
      trainable=False,
      initializer=tf.zeros_initializer()) for tvar in tvars]

    reset_step = tf.cast(tf.math.equal(local_step % num_accumulate_steps, 0), dtype=tf.bool)
    local_step = tf.cond(reset_step,
                         lambda: local_step.assign(tf.ones_like(local_step)),
                         lambda: local_step.assign_add(1))

    grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i]) for i, gv in enumerate(grads_and_vars) if
                                 gv[0] is not None]
    grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

    all_are_finite = tf.reduce_all(
      [tf.reduce_all(tf.math.is_finite(g)) for g in grads]) if use_fp16 else tf.constant(True, dtype=tf.bool)
    batch_finite = tf.cond(
      reset_step,
      lambda: batch_finite.assign(tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
      lambda: batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite)))

    accum_vars = tf.cond(reset_step,
                         lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(grads)],
                         lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(grads)])

    def update(accum_vars_):
      if allreduce_post_accumulation and hvd is not None:
        accum_vars_ = [
          hvd.allreduce(tf.convert_to_tensor(accum_var),
                        compression=hvd.Compression.fp16 if use_fp16 else hvd.Compression.none)
          if isinstance(accum_var, tf.IndexedSlices)
          else hvd.allreduce(accum_var, compression=hvd.Compression.fp16 if use_fp16 else hvd.Compression.none)
          for accum_var in accum_vars_]

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hizzy fit.
      clipped_grads, _ = tf.clip_by_global_norm(
        accum_vars_, clip_norm=1.0,
        use_norm=tf.cond(batch_finite,
                         lambda: tf.linalg.global_norm(accum_vars_),
                         lambda: tf.constant(1.0)))
      if lr_layer_decay_rate != 1.0:
        n_layer = 0
        for i in range(len(clipped_grads)):
          m = re.search(r"bert/encoder/layer_(\d+?)/", tvars[i].name)
          if not m:
            continue
          n_layer = max(n_layer, int(m.group(1)) + 1)

        for i in range(len(clipped_grads)):
          for layer in range(n_layer):
            if "bert/encoder/layer_{}/".format(layer) in tvars[i].name:
              abs_rate = lr_layer_decay_rate ** (n_layer - 1 - layer)
              clipped_grads[i] *= abs_rate
              tf.compat.v1.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
                abs_rate, layer, tvars[i].name))
              break

      return optimizer.apply_gradients(list(zip(clipped_grads, tvars)), global_step=global_step)

    update_step = tf.identity(tf.cast(tf.math.equal(local_step % num_accumulate_steps, 0), dtype=tf.bool),
                              name="update_step")
    update_op = tf.cond(update_step,
                        lambda: update(accum_vars), lambda: tf.no_op())
    global_step = tf.identity(global_step, name='step_update')
    train_op = tf.group(update_op, [global_step])
  else:
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    grads, tvars = list(zip(*grads_and_vars))
    all_are_finite = tf.reduce_all(
      [tf.reduce_all(tf.math.is_finite(g)) for g in grads]) if use_fp16 else tf.constant(True, dtype=tf.bool)

    if hvd is not None:
      grads = [
        hvd.allreduce(tf.convert_to_tensor(grad),
                      compression=hvd.Compression.fp16 if use_fp16 else hvd.Compression.none)
        if isinstance(grad, tf.IndexedSlices)
        else hvd.allreduce(grad, compression=hvd.Compression.fp16 if use_fp16 else hvd.Compression.none)
        for grad in grads]

    (clipped_grads, _) = tf.clip_by_global_norm(
      grads, clip_norm=1.0,
      use_norm=tf.cond(
        all_are_finite,
        lambda: tf.linalg.global_norm(grads),
        lambda: tf.constant(1.0)))

    if lr_layer_decay_rate != 1.0:
      n_layer = 0
      for i in range(len(clipped_grads)):
        m = re.search(r"bert/encoder/layer_(\d+?)/", tvars[i].name)
        if not m:
          continue
        n_layer = max(n_layer, int(m.group(1)) + 1)

      for i in range(len(clipped_grads)):
        for layer in range(n_layer):
          if "bert/encoder/layer_{}/".format(layer) in tvars[i].name:
            abs_rate = lr_layer_decay_rate ** (n_layer - 1 - layer)
            clipped_grads[i] *= abs_rate
            tf.compat.v1.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
              abs_rate, layer, tvars[i].name))
            break

    update_op = optimizer.apply_gradients(
      list(zip(clipped_grads, tvars)), global_step=global_step)
    global_step = tf.identity(global_step, name='step_update')
    train_op = tf.group(update_op, [global_step])

  return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, 'learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, manual_fp16=False):
    """See base class."""
    assignments = []
    steps = tf.cast(global_step + 1, tf.float32)
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        param_fp32 = tf.compat.v1.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(), tf.float32))
      else:
        param_fp32 = param

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      beta1_correction = (1 - self.beta_1 ** steps)
      beta2_correction = (1 - self.beta_2 ** steps)

      next_m_unbiased = next_m / beta1_correction
      next_v_unbiased = next_v / beta2_correction

      update = next_m_unbiased / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want to decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      update_with_lr = self.learning_rate * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        param.assign(tf.cast(next_param, param.dtype.base_dtype))
      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class AdamWeightDecayOptimizerV2(tf.compat.v1.train.Optimizer):
  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer",
               manual_fp16=False):
    super(AdamWeightDecayOptimizerV2, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, 'learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.manual_fp16 = manual_fp16
    self.learning_rate_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _prepare(self):
    self.learning_rate_t = ops.convert_to_tensor(
      self.learning_rate, name='learning_rate')
    self.weight_decay_rate_t = ops.convert_to_tensor(
      self.weight_decay_rate, name='weight_decay_rate')
    self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
    self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
    self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

  def _create_slots(self, var_list):
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self.beta_1,
                                   name="beta1_power",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self.beta_2,
                                   name="beta2_power",
                                   colocate_with=first_var)
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
      self._zeros_slot(v, 'v', self._name)

  def _apply_dense(self, grad, var):
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    var_name = self._get_variable_name(var.name)
    has_shadow = self.manual_fp16 and var.dtype.base_dtype != tf.float32
    if has_shadow:
      var_fp32 = tf.compat.v1.get_variable(
        name=var_name + "/shadow",
        dtype=tf.float32,
        trainable=False,
        initializer=tf.cast(var.initialized_value(), tf.float32))
    else:
      var_fp32 = var

    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
    learning_rate_t = (learning_rate_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    # Standard Adam update.
    next_m = (
        tf.multiply(beta_1_t, m) +
        tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
        tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                               tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var_fp32

    update_with_lr = learning_rate_t * update

    next_param = var_fp32 - update_with_lr

    if has_shadow:
      var.assign(tf.cast(next_param, var.dtype.base_dtype))

    return control_flow_ops.group(*[var_fp32.assign(next_param),
                                    m.assign(next_m),
                                    v.assign(next_v)])

  def _resource_apply_dense(self, grad, var):
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    var_name = self._get_variable_name(var.name)
    has_shadow = self.manual_fp16 and var.dtype.base_dtype != tf.float32
    if has_shadow:
      var_fp32 = tf.compat.v1.get_variable(
        name=var_name + "/shadow",
        dtype=tf.float32,
        trainable=False,
        initializer=tf.cast(var.initialized_value(), tf.float32))
    else:
      var_fp32 = var

    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
    learning_rate_t = (learning_rate_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    # Standard Adam update.
    next_m = (
        tf.multiply(beta_1_t, m) +
        tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
        tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                               tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var_fp32

    update_with_lr = learning_rate_t * update

    next_param = var_fp32 - update_with_lr

    if has_shadow:
      var.assign(tf.cast(next_param, var.dtype.base_dtype))

    return control_flow_ops.group(*[var_fp32.assign(next_param),
                                    m.assign(next_m),
                                    v.assign(next_v)])

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
    learning_rate_t = (learning_rate_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    m_t = state_ops.assign(m, m * beta_1_t,
                           use_locking=self._use_locking)

    m_scaled_g_values = grad * (1 - beta_1_t)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)

    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    var_update = state_ops.assign_sub(var,
                                      update_with_lr,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
      grad.values, var, grad.indices,
      lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
        x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
          x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
      grad, var, indices, self._resource_scatter_add)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
          beta1_power * self.beta_1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
          beta2_power * self.beta_2_t, use_locking=self._use_locking)
      return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                    name=name_scope)


class LAMBOptimizer(tf.compat.v1.train.Optimizer):
  """A LAMB optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.steps = 0

  def apply_gradients(self, grads_and_vars, global_step=None, name=None,
      manual_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        # create shadow fp32 weights for fp16 variable
        param_fp32 = tf.compat.v1.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(), tf.float32))
      else:
        param_fp32 = param

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # LAMB update
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      self.steps += 1
      beta1_correction = (1 - self.beta_1 ** self.steps)
      beta2_correction = (1 - self.beta_2 ** self.steps)

      next_m_unbiased = next_m / beta1_correction
      next_v_unbiased = next_v / beta2_correction

      update = next_m_unbiased / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want to decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      w_norm = linalg_ops.norm(param, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
          math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      update_with_lr = ratio * self.learning_rate * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        # cast shadow fp32 weights to fp16 and assign to trainable variable
        param.assign(tf.cast(next_param, param.dtype.base_dtype))
      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

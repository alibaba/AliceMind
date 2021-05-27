#!/usr/bin/env python
import tensorflow as tf
import time


class LossLoggingHook(tf.estimator.SessionRunHook):
  def __init__(self, batch_size, every_n_iter):
    self.every_n_iter = every_n_iter
    self.every_n_examples = every_n_iter * batch_size
    self.fetches = tf.estimator.SessionRunArgs(
      fetches=[
        "step_update:0",
        "total_loss:0",
        "learning_rate:0"
      ])
    self.step_start_time = -1

  def begin(self):
    self.step_start_time = time.time()

  def before_run(self, run_context):
    return self.fetches

  def after_run(self, run_context, run_values):
    global_step, total_loss, learning_rate = run_values.results
    if global_step % self.every_n_iter == 0:
      current_time = time.time()
      tf.compat.v1.logging.info(
        'global_step=%d (%.2f ex/sec) | total_loss=%2.5f | learning_rate=%.5e' % (
          global_step, self.every_n_examples / (current_time - self.step_start_time), total_loss, learning_rate))
      self.step_start_time = current_time

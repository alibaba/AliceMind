# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors.
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
import glob
import os
import shutil
import tensorflow as tf


class Checkpoint(object):
  dir = None
  file = None
  score = None
  path = None

  def __init__(self, path, score):
    self.dir = os.path.dirname(path)
    self.file = os.path.basename(path)
    self.score = score
    self.path = path


class BestCheckpointCopier(tf.estimator.Exporter):
  checkpoints = None
  checkpoints_to_keep = None
  compare_fn = None
  name = None
  score_metric = None
  sort_key_fn = None
  sort_reverse = None

  def __init__(self, name='best_checkpoints', checkpoints_to_keep=5,
               score_metric='Loss/total_loss', compare_fn=lambda x, y: x.score < y.score,
               sort_key_fn=lambda x: x.score, sort_reverse=False):
    self.checkpoints = []
    self.checkpoints_to_keep = checkpoints_to_keep
    self.compare_fn = compare_fn
    self.name = name
    self.score_metric = score_metric
    self.sort_key_fn = sort_key_fn
    self.sort_reverse = sort_reverse
    super(BestCheckpointCopier, self).__init__()

  def _copy_checkpoint(self, checkpoint):
    desination_dir = self._destination_dir(checkpoint)
    os.makedirs(desination_dir, exist_ok=True)

    for file in glob.glob(r'{}*'.format(checkpoint.path)):
      self._log('copying {} to {}'.format(file, desination_dir))
      shutil.copy(file, desination_dir)

  def _destination_dir(self, checkpoint):
    return os.path.join(checkpoint.dir, self.name)

  def _keep_checkpoint(self, checkpoint):
    self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

    self.checkpoints.append(checkpoint)
    self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)

    self._copy_checkpoint(checkpoint)

  def _log(self, statement):
    tf.compat.v1.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

  def _prune_checkpoints(self, checkpoint):
    destination_dir = self._destination_dir(checkpoint)

    for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
      self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

      old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
      for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
        self._log('removing old checkpoint file {}'.format(file))
        os.remove(file)

    self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

  def _score(self, eval_result):
    return float(eval_result[self.score_metric])

  def _should_keep(self, checkpoint):
    return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

  def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
    self._log('export checkpoint {}'.format(checkpoint_path))

    score = self._score(eval_result)
    checkpoint = Checkpoint(path=checkpoint_path, score=score)

    if self._should_keep(checkpoint):
      self._keep_checkpoint(checkpoint)
      self._prune_checkpoints(checkpoint)
    else:
      self._log('skipping checkpoint {}'.format(checkpoint.path))

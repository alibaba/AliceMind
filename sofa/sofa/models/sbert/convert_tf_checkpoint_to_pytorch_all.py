# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2020 The HuggingFace Inc. team. 
# All rights reserved.
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
"""Convert SBERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .modeling_sbert import SbertConfig, load_tf_weights_in_bert


def convert_all(tf_checkpoint_path, sbert_config_file, pytorch_dump_path, module_clz):
    """
    Convert a checkpoint from tf to pt.
    Only support backbones with a linear part called classifier.
    :param tf_checkpoint_path: The tf checkpoint local dir.
    :param sbert_config_file: The sbert config file local dir.
    :param pytorch_dump_path: The local file path of the generated pytorch bin file.
    :param module_clz: The model class.
    :return: None
    """
    # Initialise PyTorch model
    config = SbertConfig.from_json_file(sbert_config_file)
    model = module_clz(config)
    # Load weights from TF model
    load_tf_weights_in_bert(model, sbert_config_file, tf_checkpoint_path)
    # Save pytorch-model
    torch.save(model.state_dict(), pytorch_dump_path)

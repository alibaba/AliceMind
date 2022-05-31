# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
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

import os
from .compat import _report_compat_error

_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]
"""
Select the proper modeling class according to the runtime backend.
"""
if sofa_backend == "huggingface":
    import transformers
    # import transformers.onnx
    PreTrainedModel = transformers.modeling_utils.PreTrainedModel
    Application = transformers.modeling_utils.PreTrainedModel
    OnnxConfig = object
    # transformers.onnx.OnnxConfig
    apply_chunking_to_forward = transformers.modeling_utils.apply_chunking_to_forward
    find_pruneable_heads_and_indices = transformers.modeling_utils.find_pruneable_heads_and_indices
    prune_linear_layer = transformers.modeling_utils.prune_linear_layer
elif sofa_backend == "easytexminer":
    import easytexminer.applications.application
    import easytexminer.model_zoo
    PreTrainedModel = easytexminer.model_zoo.modeling_utils.PreTrainedModel
    OnnxConfig = object
    apply_chunking_to_forward = easytexminer.model_zoo.modeling_utils.apply_chunking_to_forward
    find_pruneable_heads_and_indices = easytexminer.model_zoo.modeling_utils.find_pruneable_heads_and_indices
    prune_linear_layer = easytexminer.model_zoo.modeling_utils.prune_linear_layer

    class Application(easytexminer.applications.application.Application):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            return cls(pretrained_model_name_or_path)

elif sofa_backend == "easynlp":
    import easynlp.appzoo.application
    import easynlp.modelzoo
    PreTrainedModel = easynlp.modelzoo.modeling_utils.PreTrainedModel
    OnnxConfig = object
    apply_chunking_to_forward = easynlp.modelzoo.modeling_utils.apply_chunking_to_forward
    find_pruneable_heads_and_indices = easynlp.modelzoo.modeling_utils.find_pruneable_heads_and_indices
    prune_linear_layer = easynlp.modelzoo.modeling_utils.prune_linear_layer

    class Application(easynlp.appzoo.application.Application):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            return cls(pretrained_model_name_or_path)

elif sofa_backend == "sofa":
    from .backend import modeling_utils
    # import transformers.onnx
    PreTrainedModel = modeling_utils.PreTrainedModel
    Application = modeling_utils.PreTrainedModel
    OnnxConfig = object
    # transformers.onnx.OnnxConfig
    apply_chunking_to_forward = modeling_utils.apply_chunking_to_forward
    find_pruneable_heads_and_indices = modeling_utils.find_pruneable_heads_and_indices
    prune_linear_layer = modeling_utils.prune_linear_layer

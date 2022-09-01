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
from enum import Enum
from .compat import _report_compat_error

_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]
"""
Select the proper config class according to the runtime backend.
"""
if sofa_backend == "huggingface":
    import transformers
    PretrainedConfig = transformers.configuration_utils.PretrainedConfig
elif sofa_backend == "easytexminer":
    import easytexminer.model_zoo
    PretrainedConfig = easytexminer.model_zoo.configuration_utils.PretrainedConfig
elif sofa_backend == "easynlp":
    import easynlp.modelzoo
    PretrainedConfig = easynlp.modelzoo.configuration_utils.PretrainedConfig
elif sofa_backend == "sofa":
    from .backend import configuration_utils
    PretrainedConfig = configuration_utils.PretrainedConfig


class TaskType(Enum):
    SINGLE_LABEL_CLASSIFICATION = "single_label_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    QUESTION_ANSWERING = "question_anwsering"
    GENERATION = "generation"
    SEQUENCE_LABELING = "sequence_labeling"

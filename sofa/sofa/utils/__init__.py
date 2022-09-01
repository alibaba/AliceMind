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

from .optimizer_utils import (
    ChildTuningAdamW,
    apply_child_tuning_to_trainer,
    apply_child_tuning
)
from .configuration_utils import (
    PretrainedConfig,
    TaskType
)
from .activations import (
    ACT2FN
)
from .inference_utils import (
    InferenceBase
)
from .modeling_utils import (
    PreTrainedModel,
    Application,
    OnnxConfig,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .file_utils import (
    ModelOutput,
    logging,
    add_code_sample_docstrings,
    replace_return_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_end_docstrings,
)

from .data_utils.file_utils import cached_path

from .modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    DatasetOutput,
    SpanClassifierOutput
)
from .tokenization_utils import (
    PreTrainedTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
    AddedToken,
)
from .tokenization_utils_fast import (
    PreTrainedTokenizerFast
)

from .compat import inject_model_backend, inject_pipeline

import sys

sys.path.append('./mpu')

from .utils import *
from .fp16 import *
from .data_utils import *
from .checkpoints import save_checkpoint, load_checkpoint, pre_load, load_deepspeed_checkpoint
from .args_utils import ArgsBase

from .dureader_eval import compute_bleu_rouge, normalize


# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team.
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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .file_utils import (
    _LazyModule,
    is_flax_available,
    is_pyctcdecode_available,
    is_pytorch_quantization_available,
    is_scatter_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Configuration
from .configuration_utils import PretrainedConfig

from .data_collator import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForSOP,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithPadding,
    DefaultDataCollator,
    default_data_collator,
)

# Feature Extractor
from .feature_extraction_utils import BatchFeature

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    TensorType,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_phonemizer_available,
    is_psutil_available,
    is_py3nvml_available,
    is_pyctcdecode_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_speech_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
    is_vision_available,
)

# Integrations
from .integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_tensorboard_available,
    is_wandb_available,
)

# Tokenization
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TokenSpan,
)

# Trainer
from .trainer_callback import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import EvalPrediction, IntervalStrategy, SchedulerType, set_seed
from .training_args import TrainingArguments
from .utils import logging

# Modeling

from .generation_beam_search import BeamScorer, BeamSearchScorer
from .generation_logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    LogitsWarper,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .generation_utils import top_k_top_p_filtering
from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer

# Optimization
from .optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)

# Trainer
from .trainer import Trainer
from .trainer_pt_utils import torch_distributed_zero_first
from .auto import *


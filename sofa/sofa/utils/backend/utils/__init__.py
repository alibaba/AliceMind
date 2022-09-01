#!/usr/bin/env python
# coding=utf-8

# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Inc. team.
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

from packaging import version

from .doc import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    copy_func,
    replace_return_docstrings,
)
from .generic import (
    ContextManagers,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    TensorType,
    cached_property,
    is_tensor,
    to_numpy,
    to_py_obj,
)
from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    _LazyModule,
    is_apex_available,
    is_coloredlogs_available,
    is_datasets_available,
    is_detectron2_available,
    is_faiss_available,
    is_flax_available,
    is_ftfy_available,
    is_in_notebook,
    is_librosa_available,
    is_onnx_available,
    is_pandas_available,
    is_phonemizer_available,
    is_protobuf_available,
    is_psutil_available,
    is_py3nvml_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_rjieba_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_scatter_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_soundfile_availble,
    is_spacy_available,
    is_speech_available,
    is_tensorflow_probability_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_cuda_available,
    is_torch_fx_available,
    is_torch_fx_proxy,
    is_torch_onnx_dict_inputs_support_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torchaudio_available,
    is_training_run_on_sagemaker,
    is_vision_available,
    requires_backends,
    tf_required,
    torch_only_method,
    torch_required,
    torch_version,
)


WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


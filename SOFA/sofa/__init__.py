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

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any
from pkg_resources import get_distribution
from typing import TYPE_CHECKING

try:
    __version__ = get_distribution('sofa').version
except:
    __version__ = "1.0.0.local"


class _DynamicModule(ModuleType):
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)


def environ(backend):
    os.environ["SOFA_BACKEND"] = backend
    from .models import sbert, veco, SbertConfig, SbertTokenizer, SbertTokenizerFast, SbertModel, SbertForSequenceClassification
    from .models import SbertForTokenClassification, SbertForQuestionAnswering, SbertForMultipleChoice, SbertForPreTraining
    from .models import SbertForMaskedLM, SbertForNextSentencePrediction, VecoConfig, VecoTokenizer
    from .models import VecoTokenizerFast, VecoModel, VecoForSequenceClassification, \
        VecoForMultipleChoice, VecoForTokenClassification, VecoForQuestionAnswering
    from .models import palm, PalmConfig, PalmTokenizer, PalmTokenizerFast, PalmModel, PalmForConditionalGeneration
    from .utils import inject_model_backend
    inject_model_backend("sbert", "structbert", SbertConfig, SbertTokenizer,
                         SbertTokenizerFast, backbone=SbertModel,
                         sequence_classification=SbertForSequenceClassification,
                         token_classification=SbertForTokenClassification,
                         question_answering=SbertForQuestionAnswering,
                         multiple_choice=SbertForMultipleChoice,
                         pre_train=SbertForPreTraining,
                         mlm=SbertForMaskedLM,
                         nsp=SbertForNextSentencePrediction,
                         module=sbert)
    inject_model_backend("veco", "veco", VecoConfig, VecoTokenizer,
                         VecoTokenizerFast, backbone=VecoModel,
                         sequence_classification=VecoForSequenceClassification,
                         token_classification=VecoForTokenClassification,
                         question_answering=VecoForQuestionAnswering,
                         multiple_choice=VecoForMultipleChoice,
                         slow_to_fast_converter="XLMRobertaTokenizer",
                         module=veco)
    inject_model_backend("palm", "palm", PalmConfig, PalmTokenizer,
                         PalmTokenizerFast, backbone=PalmModel,
                         s2slm=PalmForConditionalGeneration,
                         module=palm)


_import_structure = {
    "utils": [
        "ChildTuningAdamW",
        "apply_child_tuning_to_trainer",
        "apply_child_tuning",
        "PretrainedConfig",
        "TaskType",
        "ACT2FN",
        "InferenceBase",
        "PreTrainedModel",
        "Application",
        "OnnxConfig",
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
        "ModelOutput",
        "logging",
        "add_code_sample_docstrings",
        "replace_return_docstrings",
        "add_start_docstrings",
        "add_start_docstrings_to_model_forward",
        "add_end_docstrings",
        "BaseModelOutputWithPastAndCrossAttentions",
        "BaseModelOutputWithPoolingAndCrossAttentions",
        "CausalLMOutputWithCrossAttentions",
        "MaskedLMOutput",
        "MultipleChoiceModelOutput",
        "NextSentencePredictorOutput",
        "QuestionAnsweringModelOutput",
        "SequenceClassifierOutput",
        "TokenClassifierOutput",
        "DatasetOutput",
        "SpanClassifierOutput",
        "PreTrainedTokenizer",
        "_is_control",
        "_is_punctuation",
        "_is_whitespace",
        "AddedToken",
        "PreTrainedTokenizerFast",
        "check_update",
        "inject_model_backend",
        "inject_pipeline"
    ],
    "models": [
        "sbert",
        "veco",
        "palm",
        "SbertConfig",
        "SbertTokenizer",
        "SbertTokenizerFast",
        "SbertForSequenceClassification",
        "SbertModel",
        "SbertForQuestionAnswering",
        "SbertForTokenClassification",
        "SbertForMultipleChoice",
        "SbertForPreTraining",
        "SbertForMaskedLM",
        "SbertForNextSentencePrediction",
        "VecoConfig",
        "VecoModel",
        "VecoForSequenceClassification",
        "VecoForMultipleChoice",
        "VecoForQuestionAnswering",
        "VecoForTokenClassification",
        "VecoTokenizer",
        "VecoTokenizerFast",
        "PalmConfig",
        "PalmModel",
        "PalmTokenizer",
        "PalmTokenizerFast",
        "PalmForConditionalGeneration",
    ],
    "examples": [
        "run_sequence_labeling_hf",
        "run_classification_hf",
        "run_generation_hf"
    ],
    "sofa": [
        "environ"
    ]
}

if TYPE_CHECKING:
    from .models import (
        sbert,
        veco,
        palm,
        SbertConfig,
        SbertTokenizer,
        SbertTokenizerFast,
        SbertForSequenceClassification,
        SbertModel,
        SbertForQuestionAnswering,
        SbertForTokenClassification,
        SbertForMultipleChoice,
        SbertForPreTraining,
        SbertForMaskedLM,
        SbertForNextSentencePrediction,
        VecoConfig,
        VecoModel,
        VecoForSequenceClassification,
        VecoForMultipleChoice,
        VecoForQuestionAnswering,
        VecoForTokenClassification,
        VecoTokenizer,
        VecoTokenizerFast,
        PalmConfig,
        PalmModel,
        PalmTokenizer,
        PalmTokenizerFast,
        PalmForConditionalGeneration,
    )
    from .examples import (
        run_sequence_labeling_hf,
        run_classification_hf,
        run_generation_hf,
    )
    from .utils import *
else:
    import sys
    sys.modules[__name__] = _DynamicModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__, "environ": environ},
    )


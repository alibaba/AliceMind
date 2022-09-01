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
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from .file_utils import ModelOutput

from .compat import _report_compat_error

_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]
"""
Select the proper output class according to the runtime backend.
"""
if sofa_backend == "huggingface":
    import transformers.modeling_outputs
    BaseModelOutputWithPastAndCrossAttentions = \
        transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
    BaseModelOutputWithPoolingAndCrossAttentions = \
        transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    CausalLMOutputWithCrossAttentions = \
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    MaskedLMOutput = transformers.modeling_outputs.MaskedLMOutput
    MultipleChoiceModelOutput = transformers.modeling_outputs.MultipleChoiceModelOutput
    NextSentencePredictorOutput = transformers.modeling_outputs.NextSentencePredictorOutput
    QuestionAnsweringModelOutput = transformers.modeling_outputs.QuestionAnsweringModelOutput
    SequenceClassifierOutput = transformers.modeling_outputs.SequenceClassifierOutput
    TokenClassifierOutput = transformers.modeling_outputs.TokenClassifierOutput
elif sofa_backend == "easytexminer":
    import easytexminer.model_zoo.modeling_outputs
    BaseModelOutputWithPastAndCrossAttentions = \
        easytexminer.model_zoo.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
    BaseModelOutputWithPoolingAndCrossAttentions = \
        easytexminer.model_zoo.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    CausalLMOutputWithCrossAttentions = \
        easytexminer.model_zoo.modeling_outputs.CausalLMOutputWithCrossAttentions
    MaskedLMOutput = easytexminer.model_zoo.modeling_outputs.MaskedLMOutput
    MultipleChoiceModelOutput = easytexminer.model_zoo.modeling_outputs.MultipleChoiceModelOutput
    NextSentencePredictorOutput = easytexminer.model_zoo.modeling_outputs.NextSentencePredictorOutput
    QuestionAnsweringModelOutput = easytexminer.model_zoo.modeling_outputs.QuestionAnsweringModelOutput
    SequenceClassifierOutput = easytexminer.model_zoo.modeling_outputs.SequenceClassifierOutput
    TokenClassifierOutput = easytexminer.model_zoo.modeling_outputs.TokenClassifierOutput
elif sofa_backend == "easynlp":
    import easynlp.modelzoo.modeling_outputs
    BaseModelOutputWithPastAndCrossAttentions = \
        easynlp.modelzoo.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
    BaseModelOutputWithPoolingAndCrossAttentions = \
        easynlp.modelzoo.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    CausalLMOutputWithCrossAttentions = \
        easynlp.modelzoo.modeling_outputs.CausalLMOutputWithCrossAttentions
    MaskedLMOutput = easynlp.modelzoo.modeling_outputs.MaskedLMOutput
    MultipleChoiceModelOutput = easynlp.modelzoo.modeling_outputs.MultipleChoiceModelOutput
    NextSentencePredictorOutput = easynlp.modelzoo.modeling_outputs.NextSentencePredictorOutput
    QuestionAnsweringModelOutput = easynlp.modelzoo.modeling_outputs.QuestionAnsweringModelOutput
    SequenceClassifierOutput = easynlp.modelzoo.modeling_outputs.SequenceClassifierOutput
    TokenClassifierOutput = easynlp.modelzoo.modeling_outputs.TokenClassifierOutput
elif sofa_backend == "sofa":
    from .backend import modeling_outputs
    BaseModelOutputWithPastAndCrossAttentions = \
        modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
    BaseModelOutputWithPoolingAndCrossAttentions = \
        modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    CausalLMOutputWithCrossAttentions = \
        modeling_outputs.CausalLMOutputWithCrossAttentions
    MaskedLMOutput = modeling_outputs.MaskedLMOutput
    MultipleChoiceModelOutput = modeling_outputs.MultipleChoiceModelOutput
    NextSentencePredictorOutput = modeling_outputs.NextSentencePredictorOutput
    QuestionAnsweringModelOutput = modeling_outputs.QuestionAnsweringModelOutput
    SequenceClassifierOutput = modeling_outputs.SequenceClassifierOutput
    TokenClassifierOutput = modeling_outputs.TokenClassifierOutput


@dataclass
class DatasetOutput(object):
    sent1: str = None
    sent2: str = None
    label: int = None


@dataclass
class SpanClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    
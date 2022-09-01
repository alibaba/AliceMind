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
from torch import nn
from .compat import _report_compat_error
from .configuration_utils import TaskType
from .file_utils import logging
from packaging import version

logger = logging.get_logger(__name__)
_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]
"""
Select the proper Predictor class according to the runtime backend.
"""
if sofa_backend == "huggingface":
    import transformers
    Predictor = transformers.pipelines.Pipeline
elif sofa_backend == "easytexminer":
    import easytexminer.core.predictor as predictor
    Predictor = predictor.Predictor
elif sofa_backend == "easynlp":
    import easynlp.core.predictor as predictor
    Predictor = predictor.Predictor
elif sofa_backend == "sofa":
    Predictor = object


class InferenceBase(Predictor):
    """
    Inference Base class, Inherited from transformers.Pipeline/easynlp.Predictor.
    """

    def __init__(self,
                 model: nn.Module,
                 tokenizer=None,
                 task: str = "",
                 device: int = -1,
                 *args, **kwargs):
        """
        Init bound method.
        :param model: The actual model, should be nn.Module type.
        :param tokenizer: The actual tokenizer.
        :param task: The task type which will be passed from Pipeline.
        :param device: Device type.
        :param args: Extra args.
        :param kwargs: Extra kwargs.
        """
        if model is None:
            raise RuntimeError("Sofa does not support pipeline with default model type, please specify one")
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"Input model should be a sub class of nn.Module")
        if sofa_backend == "huggingface":
            # from version 4.11.0, transformers changed its predict behavior:
            # add _sanitize_parameters to split pre/forward/post params
            # and totally changed the class code.
            if version.parse(transformers.__version__) < version.parse("4.11.0"):
                super().__init__(model=model, tokenizer=tokenizer, task=task, device=device,
                                 *args, **kwargs)
                logger.warning(f"transformers version is:{transformers.__version__}, "
                               f"extra kwargs will not be used in _sanitize_parameters")
                self._preprocess_params = {}
                self._forward_params = {}
                self._postprocess_params = {}
            else:
                super().__init__(model=model, tokenizer=tokenizer, task=task, device=device,
                                 *args, **kwargs)
        else:
            if sofa_backend == "easytexminer":
                super().__init__(*args, **kwargs)
            assert tokenizer is not None
            self.model = model
            self.model.eval()
            self.tokenizer = tokenizer
            self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)

            task_specific_params = self.model.config.task_specific_params
            if task_specific_params is not None and task in task_specific_params:
                self.model.config.update(task_specific_params.get(task))
            self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(
                **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        # Split pipeline_parameters to preprocessing parameters, predict parameters,
        # postprocessing parameters.
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def preprocess(self, inputs, **kwargs):
        """
        The preprocess method.
        :param inputs: Actual sentense input.
        :param kwargs: Any extra kwargs, passed from _sanitize_parameters of self.__init__, or
        actual calling.
        :return: The preprocess data result.
        """
        raise NotImplementedError

    def _forward(self, inputs, **kwargs):
        return self.predict(inputs, **kwargs)

    def predict(self, inputs, **kwargs):
        """
        The predict method.
        :param inputs: Actual input from preprocess.
        :param kwargs: Any extra kwargs, passed from _sanitize_parameters of self.__init__, or
        actual calling.
        :return: The predict data result.
        """
        raise NotImplementedError

    def postprocess(self, outputs, **kwargs):
        """
        The postprocess method.
        :param outputs: Actual input from predict.
        :param kwargs: Any extra kwargs, passed from _sanitize_parameters of self.__init__, or
        actual calling.
        :return: The postprocess data result.
        """
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        if sofa_backend == "huggingface":
            import transformers
            if version.parse(transformers.__version__) < version.parse("4.11.0"):
                return self._call_default(inputs, **kwargs)
            else:
                return super().__call__(inputs, **kwargs)
        elif sofa_backend == "easytexminer":
            return super().run(inputs)
        elif sofa_backend == "sofa":
            return self._call_default(inputs, **kwargs)
        else:
            _report_compat_error()

    def run(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def _call_default(self, inputs, **kwargs):
        """
        A default flow when version not match or no outer backend.
        :param inputs: Actual sentense input.
        :param kwargs: Any extra kwargs, passed from _sanitize_parameters of self.__init__, or
        actual calling.
        :return: The processed data result.
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.predict(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    @classmethod
    def from_pretrained(cls, model, env, *args, **kwargs):
        return cls(model, *args, **kwargs)


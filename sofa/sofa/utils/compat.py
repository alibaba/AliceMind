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

import importlib
import os

from packaging import version

_registered_modules = set()
supported_min_version = "4.10.0"
supported_max_version = "4.18.0"


def _report_compat_error():
    if "SOFA_BACKEND" not in os.environ:
        os.environ["SOFA_BACKEND"] = "sofa"

    sofa_backend = os.environ["SOFA_BACKEND"]
    if sofa_backend not in ["huggingface", "easytexminer", "easynlp", "sofa"]:
        raise RuntimeError(f"Sofa backend {sofa_backend} not supported.")


def inject_pipeline(name, pipeline_clz, automodel_clz):
    """
    Inject custom pipeline into transformers pipeline
    :param name: The pipeline name
    :param pipeline_clz: The pipeline clz, should be the sub class of InferenceBase
    :param automodel_clz: The AutoModel class to get the proper model from.
    :return: None
    """
    _report_compat_error()
    sofa_backend = os.environ["SOFA_BACKEND"]
    if sofa_backend == "huggingface":
        import transformers.pipelines as pipelines
        pipelines.SUPPORTED_TASKS[name] = {
            "impl": pipeline_clz,
            "tf": (),
            "pt": (automodel_clz,),
            "default": {},
        }


def inject_model_backend(name, full_name, config, tokenizer, tokenizer_fast, **kwargs):
    """
    Inject some model package into the selected backend framework.
    :param name: The model name.
    :param full_name: The full name of the model
    :param config: The config class of the model.
    :param tokenizer: The tokenizer class of the model.
    :param tokenizer_fast: The tokenizer fast class of the model.
    :param kwargs: The specific task class of the model.
    Supported:
    backbone: The backbone model class
    sequence_classification: The sequence classfication model class
    token_classification: The token classification model class
    question_answering: The question-answering model class
    multiple_choice: The multiple choice(e.g. SWAG task) model class
    pre_train: The pretrain model class
    mlm: The Masked language model class
    nsp: The nsp model class
    module: The module package
    :return: None
    """
    _report_compat_error()
    sofa_backend = os.environ["SOFA_BACKEND"]
    if sofa_backend == "huggingface":
        _huggingface(name, full_name, config, tokenizer, tokenizer_fast, **kwargs)
    elif sofa_backend in ["easytexminer", "easynlp"]:
        _easyx(name, full_name, config, tokenizer, tokenizer_fast, **kwargs)


def _load_attr_from_module_with_extra_modules(self, model_type, attr):
    """
    Replace load module implementation of hf.
    This function will firstly search from sofa.models, then search from transformers if not found.
    :param self: self
    :param model_type: A str type model name
    :param attr: Any attr from the module.
    :return: The loaded attr.
    """
    from transformers.models.auto import auto_factory
    if model_type in _registered_modules:
        if model_type not in self._modules:
            self._modules[model_type] = importlib.import_module(f".{model_type}", "sofa.models")
        return auto_factory.getattribute_from_module(self._modules[model_type], attr)
    return self._load_attr_from_module_local(model_type, attr)


def _huggingface(name, full_name, config, tokenizer, tokenizer_fast, slow_to_fast_converter="BertTokenizer", **kwargs):
    """
    Register a model to hf.
    :param name: The model name.
    :param full_name: The full name of the model
    :param config: The config class of the model.
    :param tokenizer: The tokenizer class of the model.
    :param tokenizer_fast: The tokenizer fast class of the model.
    :param slow_to_fast_converter: The slow_to_fast_converter.
    :param kwargs: The specific task class of the model.
    Supported:
    backbone: The backbone model class
    sequence_classification: The sequence classfication model class
    token_classification: The token classification model class
    question_answering: The question-answering model class
    multiple_choice: The multiple choice(e.g. SWAG task) model class
    pre_train: The pretrain model class
    mlm: The Masked language model class
    nsp: The nsp model class
    module: The module package
    :return: None
    """
    import transformers
    from transformers.models.auto import configuration_auto
    from transformers.models.auto import auto_factory
    from transformers.models.auto import modeling_auto
    from transformers.models.auto import tokenization_auto
    from transformers import \
        AutoModelForSequenceClassification, \
        AutoModelForTokenClassification, \
        AutoModelForPreTraining, \
        AutoModelForQuestionAnswering, \
        AutoModelForMaskedLM, \
        AutoModelForSeq2SeqLM, \
        AutoModelForMultipleChoice, \
        AutoModelForNextSentencePrediction, \
        AutoModel

    task_type_mapping = {
        "sequence_classification": AutoModelForSequenceClassification,
        "token_classification": AutoModelForTokenClassification,
        "question_answering": AutoModelForQuestionAnswering,
        "multiple_choice": AutoModelForMultipleChoice,
        "pre_train": AutoModelForPreTraining,
        "mlm": AutoModelForMaskedLM,
        "s2slm": AutoModelForSeq2SeqLM,
        "nsp": AutoModelForNextSentencePrediction,
        "backbone": AutoModel,
    }

    if version.parse(transformers.__version__) < version.parse(supported_min_version):
        print(f"Warning: Your transformers version is {transformers.__version__}, lower than we asked, "
              f"the initialization of the framework may possibly be failed, please upgrade your version to "
              f"at least {supported_min_version} or contact the maintainer of this framework.")
    elif version.parse(transformers.__version__) > version.parse(supported_max_version):
        print(f"Warning: Your transformers version is {transformers.__version__}, greater than we tested yet, "
              f"if anything goes wrong, please contact the maintainer of this framework.")
    config_name = config.__name__.split(".")[-1]
    tokenizer_name = tokenizer.__name__.split(".")[-1]
    tokenizer_fast_name = tokenizer_fast.__name__.split(".")[-1]
    configuration_auto.CONFIG_MAPPING_NAMES[name] = config_name
    configuration_auto.CONFIG_MAPPING_NAMES.move_to_end(name, last=False)
    configuration_auto.MODEL_NAMES_MAPPING[name] = full_name
    configuration_auto.MODEL_NAMES_MAPPING.move_to_end(name, last=False)
    configuration_auto.CONFIG_MAPPING._mapping[name] = config_name
    configuration_auto.CONFIG_MAPPING._mapping.move_to_end(name, last=False)
    configuration_auto.CONFIG_MAPPING._modules[name] = kwargs["module"]

    if auto_factory._LazyAutoMapping._load_attr_from_module != _load_attr_from_module_with_extra_modules:
        auto_factory._LazyAutoMapping._load_attr_from_module_local = \
            auto_factory._LazyAutoMapping._load_attr_from_module
        auto_factory._LazyAutoMapping._load_attr_from_module = _load_attr_from_module_with_extra_modules

    tokenization_auto.TOKENIZER_MAPPING_NAMES[name] = (tokenizer_name, tokenizer_fast_name)
    tokenization_auto.TOKENIZER_MAPPING_NAMES.move_to_end(name, last=False)

    from transformers.models.auto.auto_factory import _LazyAutoMapping

    def _register(maper: _LazyAutoMapping, config_mapping, model_mapping):
        maper._config_mapping = config_mapping
        maper._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        maper._model_mapping = model_mapping

    if version.parse(transformers.__version__) < version.parse("4.12.0"):
        _register(tokenization_auto.TOKENIZER_MAPPING, configuration_auto.CONFIG_MAPPING_NAMES,
                  tokenization_auto.TOKENIZER_MAPPING_NAMES)
    else:
        tokenization_auto.TOKENIZER_MAPPING.register(config, (tokenizer, tokenizer_fast))
    transformers.SLOW_TO_FAST_CONVERTERS[tokenizer_name] = transformers.SLOW_TO_FAST_CONVERTERS[slow_to_fast_converter]

    def auto_inject_task_class(task, modeling_auto_name):
        if task not in kwargs:
            return
        task_class = kwargs[task]
        class_name = task_class.__name__.split(".")[-1]
        modeling_auto_name[name] = class_name
        modeling_auto_name.move_to_end(name, last=False)
        if version.parse(transformers.__version__) < version.parse("4.12.0"):
            _register(task_type_mapping[task]._model_mapping, configuration_auto.CONFIG_MAPPING_NAMES,
                      modeling_auto_name)
        else:
            task_type_mapping[task]._model_mapping.register(config, task_class)

    auto_inject_task_class("backbone",
                           modeling_auto.MODEL_MAPPING_NAMES)
    auto_inject_task_class("sequence_classification",
                           modeling_auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)
    auto_inject_task_class("token_classification",
                           modeling_auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
    auto_inject_task_class("question_answering",
                           modeling_auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)
    auto_inject_task_class("multiple_choice",
                           modeling_auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
    auto_inject_task_class("pre_train",
                           modeling_auto.MODEL_FOR_PRETRAINING_MAPPING_NAMES)
    auto_inject_task_class("mlm",
                           modeling_auto.MODEL_FOR_MASKED_LM_MAPPING_NAMES)
    auto_inject_task_class("s2slm",
                           modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
    auto_inject_task_class("nsp",
                           modeling_auto.MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES)

    global _registered_modules
    _registered_modules.add(name)


def _easyx(name, full_name, config, tokenizer, tokenizer_fast,
           slow_to_fast_converter="BertTokenizer", **kwargs):
    """
    Register a model to easyx.
    :param name: The model name.
    :param full_name: The full name of the model
    :param config: The config class of the model.
    :param tokenizer: The tokenizer class of the model.
    :param tokenizer_fast: The tokenizer fast class of the model.
    :param slow_to_fast_converter: The slow_to_fast_converter.
    :param kwargs: The specific task class of the model.
    Supported:
    backbone: The backbone model class
    sequence_classification: The sequence classfication model class
    token_classification: The token classification model class
    question_answering: The question-answering model class
    multiple_choice: The multiple choice(e.g. SWAG task) model class
    pre_train: The pretrain model class
    mlm: The Masked language model class
    nsp: The nsp model class
    module: The module package
    :return: None
    """
    sofa_backend = os.environ["SOFA_BACKEND"]
    if sofa_backend == "easytexminer":
        import easytexminer.model_zoo as modelzoo
        import easytexminer.model_zoo.tokenization_utils_fast
        from easytexminer.model_zoo.models import auto
        from easytexminer.model_zoo.models.auto import configuration_auto
        from easytexminer.model_zoo.models.auto import tokenization_auto
    else:
        import easynlp.modelzoo as modelzoo
        import easynlp.modelzoo.tokenization_utils_fast
        from easynlp.modelzoo.models import auto
        from easynlp.modelzoo.models.auto import configuration_auto
        from easynlp.modelzoo.models.auto import tokenization_auto

    tokenizer_name = tokenizer.__name__.split(".")[-1]
    configuration_auto.CONFIG_MAPPING[name] = config
    configuration_auto.CONFIG_MAPPING.move_to_end(name, last=False)
    configuration_auto.MODEL_NAMES_MAPPING[name] = full_name
    configuration_auto.MODEL_NAMES_MAPPING.move_to_end(name, last=False)
    tokenization_auto.TOKENIZER_MAPPING[config] = (tokenizer, tokenizer_fast)
    modelzoo.SLOW_TO_FAST_CONVERTERS[tokenizer_name] = modelzoo.SLOW_TO_FAST_CONVERTERS[slow_to_fast_converter]
    auto.MODEL_MAPPING[config] = kwargs["backbone"]
    if "sequence_classification" in kwargs:
        auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[config] = kwargs["sequence_classification"]
    if "token_classification" in kwargs:
        auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[config] = kwargs["token_classification"]
    if "question_answering" in kwargs:
        auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING[config] = kwargs["question_answering"]
    if "multiple_choice" in kwargs:
        auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING[config] = kwargs["multiple_choice"]
    if "pre_train" in kwargs:
        auto.MODEL_FOR_PRETRAINING_MAPPING[config] = kwargs["pre_train"]
    if "mlm" in kwargs:
        auto.MODEL_FOR_MASKED_LM_MAPPING[config] = kwargs["mlm"]
    if "s2slm" in kwargs:
        auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[config] = kwargs["s2slm"]
    if "nsp" in kwargs:
        auto.MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING[config] = kwargs["nsp"]


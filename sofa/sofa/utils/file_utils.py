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

import re
import os
import sys
from .compat import _report_compat_error

_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]
"""
Select the proper logging and output class according to the runtime backend.
"""
if sofa_backend == "huggingface":
    import transformers
    ModelOutput = transformers.modeling_utils.ModelOutput
    logging = transformers.logging
    is_sentencepiece_available = transformers.is_sentencepiece_available
elif sofa_backend == "easytexminer":
    import easytexminer.model_zoo
    ModelOutput = easytexminer.model_zoo.modeling_utils.ModelOutput
    logging = easytexminer.model_zoo.file_utils.logging
    is_sentencepiece_available = easytexminer.model_zoo.is_sentencepiece_available
elif sofa_backend == "easynlp":
    import easynlp.modelzoo
    ModelOutput = easynlp.modelzoo.modeling_utils.ModelOutput
    logging = easynlp.modelzoo.file_utils.logging
    is_sentencepiece_available = easynlp.modelzoo.is_sentencepiece_available
elif sofa_backend == "sofa":
    from .backend import modeling_utils, logging, is_sentencepiece_available
    ModelOutput = modeling_utils.ModelOutput
    is_sentencepiece_available = is_sentencepiece_available


"""Below code from huggingface"""
PT_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`torch.FloatTensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising
        various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`tf.Tensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising various
        elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""

PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='pt')
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import torch

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> import torch
        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> from datasets import load_dataset

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> # audio file is decoded on the fly
        >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_SPEECH_CTC_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> from datasets import load_dataset
        >>> import torch

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> # audio file is decoded on the fly
        >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        >>> logits = model(**inputs).logits
        >>> predicted_ids = torch.argmax(logits, dim=-1)

        >>> # transcribe speech
        >>> transcription = processor.batch_decode(predicted_ids)

        >>> # compute loss
        >>> with processor.as_target_processor():
        ...     inputs["labels"] = processor(dataset[0]["text"], return_tensors="pt").input_ids

        >>> loss = model(**inputs).loss
"""

PT_SPEECH_SEQ_CLASS_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> from datasets import load_dataset
        >>> import torch

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> feature_extractor = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> # audio file is decoded on the fly
        >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt")
        >>> logits = model(**inputs).logits
        >>> predicted_class_ids = torch.argmax(logits, dim=-1)
        >>> predicted_label = model.config.id2label[predicted_class_ids]

        >>> # compute loss - target_label is e.g. "down"
        >>> target_label = model.config.id2label[0]
        >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
        >>> loss = model(**inputs).loss
"""

# TODO
PT_SPAN_CLASS_SAMPLE = r"""
    Example::
    TODO
"""

PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": PT_SPEECH_CTC_SAMPLE,
    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    "SpanClassification": PT_SPAN_CLASS_SAMPLE,
}


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> outputs = model(input_dict)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
        >>> inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs.logits
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors='jax', padding=True)
        >>> outputs = model(**{{k: v[None, :] for k,v in encoding.items()}})

        >>> logits = outputs.logits
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from sofa import {processor_class}, {model_class}

        >>> tokenizer = {processor_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        >>> outputs = model(**inputs)

        >>> # retrieve logts for next token
        >>> next_token_logits = outputs.logits[:, -1]
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}


def add_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask=None,
    model_cls=None,
    modality=None
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(".")[0] if model_cls is None else model_cls

        if model_class[:2] == "TF":
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == "Flax":
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS

        doc_kwargs = dict(model_class=model_class, processor_class=processor_class, checkpoint=checkpoint)

        if "SequenceClassification" in model_class and modality == "audio":
            code_sample = sample_docstrings["AudioClassification"]
        elif "SequenceClassification" in model_class:
            code_sample = sample_docstrings["SequenceClassification"]
        elif "QuestionAnswering" in model_class:
            code_sample = sample_docstrings["QuestionAnswering"]
        elif "TokenClassification" in model_class:
            code_sample = sample_docstrings["TokenClassification"]
        elif "MultipleChoice" in model_class:
            code_sample = sample_docstrings["MultipleChoice"]
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            doc_kwargs["mask"] = "[MASK]" if mask is None else mask
            code_sample = sample_docstrings["MaskedLM"]
        elif "LMHead" in model_class or "CausalLM" in model_class:
            code_sample = sample_docstrings["LMHead"]
        elif "CTC" in model_class:
            code_sample = sample_docstrings["CTC"]
        elif "Model" in model_class and modality == "audio":
            code_sample = sample_docstrings["SpeechBaseModel"]
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = sample_docstrings["BaseModel"]
        elif "Span" in model_class:
            code_sample = sample_docstrings["SpanClassification"]
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        output_doc = _prepare_output_docstrings(output_type, config_class) if output_type is not None else ""
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = (fn.__doc__ or "") + "".join(docstr) + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f":class:`~transformers.{fn.__qualname__.split('.')[0]}`"
        intro = f"   The {class_name} forward method, overrides the :func:`__call__` special method."
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within this function, one should call the
        :class:`Module` instance afterwards instead of this since the former takes care of running the pre and post
        processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn

    return docstring_decorator


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])
        docstrings = _convert_output_args_doc(docstrings)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings

def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])
        docstrings = _convert_output_args_doc(docstrings)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings

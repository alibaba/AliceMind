# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes"""

from ..sbert.tokenization_sbert_fast import SbertTokenizerFast
from ...utils.file_utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {}
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "chinese-palm-base": 512,
    "chinese-palm-lite": 512,
    "chinese-palm-tiny": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "chinese-palm-base": {},
    "chinese-palm-lite": {},
    "chinese-palm-tiny": {},
}

class PalmTokenizerFast(SbertTokenizerFast):
    r"""
    Construct a "fast" Palm tokenizer (backed by HuggingFace's *tokenizers* library).

    [`PalmTokenizerFast`] is identical to [`SbertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`SbertTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
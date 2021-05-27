# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List
from dataclasses import dataclass, field
from tokenizers import BertWordPieceTokenizer
from pygtrie import StringTrie
from tokenization import convert_by_vocab
import copy
import igraph
import tensorflow as tf
import bisect


@dataclass
class LatticeEncoding(object):
  tokens: List[str] = field(default_factory=list)
  lengths: List[int] = field(default_factory=list)
  positions: List[int] = field(default_factory=list)
  cand_indices: List[List[int]] = field(default_factory=list)
  _to_pop_front = None
  _to_pop_back = None

  def extend(self, other):
    assert isinstance(other, LatticeEncoding)

    if len(self.tokens) == 0:
      offset = 0
    else:
      offset = max((position + length)
                   for position, length in zip(self.positions, self.lengths))

    self.lengths.extend(other.lengths)
    for position in other.positions:
      self.positions.append(position + offset)
    self.tokens.extend(other.tokens)

    if self.cand_indices is not None:
      assert other.cand_indices is not None
      for cand_index in other.cand_indices:
        new_cand_index = [index + offset for index in cand_index]
        self.cand_indices.append(new_cand_index)

  def pop_front(self):
    self.tokens = self.tokens[1:]
    self.lengths = self.lengths[1:]
    self.positions = self.positions[1:]

    min_position = min(self.positions)
    self.positions = [position - min_position for position in self.positions]

    if self.cand_indices is not None:
      new_cand_indices = []
      for cand_index in self.cand_indices:
        new_cand_index = [index - 1 for index in cand_index if index > 0]
        if len(new_cand_index) > 0:
          new_cand_indices.append(new_cand_index)
      self.cand_indices = new_cand_indices

  def pop_back(self):
    last = len(self.tokens) - 1
    self.tokens.pop()
    self.lengths.pop()
    self.positions.pop()

    if self.cand_indices is not None:
      new_cand_indices = []
      for cand_index in self.cand_indices:
        new_cand_index = [index for index in cand_index if index != last]
        if len(new_cand_index) > 0:
          new_cand_indices.append(new_cand_index)
      self.cand_indices = new_cand_indices

  def lazy_length(self):
    length = len(self.tokens)
    if self._to_pop_back:
      length -= self._to_pop_back
    if self._to_pop_front:
      length -= self._to_pop_front
    return length

  def lazy_pop_front(self):
    if self._to_pop_front is None:
      self._to_pop_front = 0
    self._to_pop_front += 1

  def lazy_pop_back(self):
    if self._to_pop_back is None:
      self._to_pop_back = 0
    self._to_pop_back += 1

  def finalize_lazy_pop_front(self):
    if self._to_pop_front is None:
      return

    self.tokens = self.tokens[self._to_pop_front:]
    self.lengths = self.lengths[self._to_pop_front:]
    self.positions = self.positions[self._to_pop_front:]

    min_position = min(self.positions)
    self.positions = [position - min_position for position in self.positions]

    if self.cand_indices is not None:
      new_cand_indices = []
      for cand_index in self.cand_indices:
        new_cand_index = [index - self._to_pop_front for index in cand_index if index >= self._to_pop_front]
        if len(new_cand_index) > 0:
          new_cand_indices.append(new_cand_index)
      self.cand_indices = new_cand_indices
    self._to_pop_front = None

  def finalize_lazy_pop_back(self):
    if self._to_pop_back is None:
      return
    self.tokens = self.tokens[:-self._to_pop_back]
    self.lengths = self.lengths[:-self._to_pop_back]
    self.positions = self.positions[:-self._to_pop_back]

    if self.cand_indices is not None:
      new_cand_indices = []
      boundary = len(self.tokens) - self._to_pop_back
      for cand_index in self.cand_indices:
        new_cand_index = [index for index in cand_index if index < boundary]
        if len(new_cand_index) > 0:
          new_cand_indices.append(new_cand_index)
      self.cand_indices = new_cand_indices
    self._to_pop_back = None

  def pop_front_chunk(self):
    raise NotImplementedError()

  def pop_back_chunk(self):
    raise NotImplementedError()

  def position_embedding(self, modes):
    assert all([mode in ('start', 'middle', 'end', 'length') for mode in modes])

    retval = []
    for mode in modes:
      if mode == 'start':
        retval.append([t for t in self.positions])
      elif mode == 'length':
        retval.append([t for t in self.lengths])
      elif mode == 'end':
        retval.append([s + l - 1 for s, l in zip(self.positions, self.lengths)])
      elif mode == 'middle':
        retval.append([s + s + l - 1 for s, l in zip(self.positions, self.lengths)])
      else:
        assert False

    return retval


class LatticeTokenizer(object):
  kDelimiter = '\u0001'

  def __init__(self,
               vocab_file,
               lexicon_file,
               do_lower_case):
    self.bert_tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=do_lower_case)
    self.lexicon = StringTrie(separator=self.kDelimiter)

    self.vocab = copy.copy(self.bert_tokenizer.get_vocab())
    # absent -> ab #sent
    # key = ab\u0001#sent
    # zzz -> z #z #z
    # lexicon = {zzz}
    # vocab = {z, #z}
    with open(lexicon_file, 'r') as reader:
      for lid, line in enumerate(reader):
        output = self.bert_tokenizer.encode(line, add_special_tokens=False)
        key = self.kDelimiter.join(output.tokens)
        name = line.strip()
        self.lexicon[key] = (name, len(output.tokens))
        self.vocab[name] = len(self.vocab)

    tf.compat.v1.logging.info(f"number of lexicon entries: {len(self.lexicon)}")
    tf.compat.v1.logging.info(f"number of vocab entries: {len(self.vocab)}")
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

    self.unk_token = self.bert_tokenizer._parameters["unk_token"]
    self.unk_token_id = self.bert_tokenizer.token_to_id(self.unk_token)
    assert self.unk_token_id is not None

    self.cls_token = self.bert_tokenizer._parameters["cls_token"]
    self.cls_token_id = self.bert_tokenizer.token_to_id(self.cls_token)
    assert self.cls_token_id is not None

    self.sep_token = self.bert_tokenizer._parameters["sep_token"]
    self.sep_token_id = self.bert_tokenizer.token_to_id(self.sep_token)
    assert self.sep_token_id is not None

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(vocab=self.vocab, items=tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(vocab=self.inv_vocab, items=ids)

  def tokenize(self, text, add_candidate_indices=False):
    output = self.bert_tokenizer.encode(text, add_special_tokens=False)
    payload = [(token, index, 1) for index, token in enumerate(output.tokens)]

    for i in range(len(output.tokens)):
      for key, (name, length) in self.lexicon.prefixes(self.kDelimiter.join(output.tokens[i:])):
        payload.append((name, i, length))

    tokens = []
    positions = []
    lengths = []
    for token, position, length in sorted(payload, key=lambda x: (x[1] + x[2], x[2])):
      tokens.append(token)
      positions.append(position)
      lengths.append(length)

    if add_candidate_indices:
      n_vertices = len(output.tokens)

      graph = igraph.Graph(n=n_vertices + 1,
                           edges=[(position, position + length) for (position, length) in zip(positions, lengths)],
                           directed=True)
      boundaries = []
      s, t = 0, n_vertices
      for n in range(1, n_vertices):
        edges = []
        edges.extend(
          [(neighbor, n) for neighbor in graph.predecessors(n)])
        edges.extend(
          [(n, neighbor) for neighbor in graph.successors(n)])
        graph.delete_edges(edges)
        path_length = graph.shortest_paths(s, t)[0][0]
        if path_length == float("inf"):
          boundaries.append(n)
        graph.add_edges(edges)

      boundaries.sort()
      boundaries = boundaries + [len(tokens)]
      cand_indices = [[] for _ in boundaries]
      for index, (position, length) in enumerate(zip(positions, lengths)):
        right_index = bisect.bisect_left(boundaries, position + length)
        cand_indices[right_index].append(index)
    else:
      cand_indices = None

    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)

  def build_cls_encoding(self, add_candidate_indices=True):
    tokens = [self.cls_token]
    positions = [0]
    lengths = [1]
    cand_indices = [] if add_candidate_indices else None
    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)

  def build_sep_encoding(self, add_candidate_indices=True):
    tokens = [self.sep_token]
    positions = [0]
    lengths = [1]
    cand_indices = [] if add_candidate_indices else None
    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)


class LatticeTokenizerWithMapping(object):
  kDelimiter = '\u0001'

  def __init__(self,
               vocab_file,
               lexicon_file,
               do_lower_case):
    self.bert_tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=do_lower_case)
    self.lexicon = StringTrie(separator=self.kDelimiter)

    offset = self.bert_tokenizer.get_vocab_size()
    self.vocab = copy.copy(self.bert_tokenizer.get_vocab())
    # absent -> ab #sent
    # key = ab\u0001#sent
    # zzz -> z #z #z
    # lexicon = {zzz}
    # vocab = {z, #z}
    unique_lexicon_entry_to_ids = {}
    with open(lexicon_file, 'r') as reader:
      for line in reader:
        fields = line.strip().split('\t')
        if fields[-1] not in unique_lexicon_entry_to_ids:
          unique_lexicon_entry_to_ids[fields[-1]] = len(unique_lexicon_entry_to_ids)

    tf.compat.v1.logging.info(f"number of lexicon entries {len(unique_lexicon_entry_to_ids)}")
    self.vocab_words_ = list(self.bert_tokenizer.get_vocab())
    with open(lexicon_file, 'r') as reader:
      for line in reader:
        fields = line.strip().split('\t')
        output = self.bert_tokenizer.encode(fields[0], add_special_tokens=False)
        key = self.kDelimiter.join(output.tokens)
        self.lexicon[key] = fields[-1], len(output.tokens)
        self.vocab[fields[-1]] = unique_lexicon_entry_to_ids[fields[-1]] + offset

    # inverse is impossible.
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

    self.unk_token = self.bert_tokenizer._parameters["unk_token"]
    self.unk_token_id = self.bert_tokenizer.token_to_id(self.unk_token)
    assert self.unk_token_id is not None

    self.cls_token = self.bert_tokenizer._parameters["cls_token"]
    self.cls_token_id = self.bert_tokenizer.token_to_id(self.cls_token)
    assert self.cls_token_id is not None

    self.sep_token = self.bert_tokenizer._parameters["sep_token"]
    self.sep_token_id = self.bert_tokenizer.token_to_id(self.sep_token)
    assert self.sep_token_id is not None

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(vocab=self.vocab, items=tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(vocab=self.inv_vocab, items=ids)

  def tokenize(self, text, add_candidate_indices=True):
    output = self.bert_tokenizer.encode(text, add_special_tokens=False)
    payload = [(token, index, 1) for index, token in enumerate(output.tokens)]

    for i in range(len(output.tokens)):
      for key, (name, length) in self.lexicon.prefixes(self.kDelimiter.join(output.tokens[i:])):
        payload.append((name, i, length))

    tokens = []
    positions = []
    lengths = []
    for token, position, length in sorted(payload, key=lambda x: (x[1] + x[2], x[2])):
      tokens.append(token)
      positions.append(position)
      lengths.append(length)

    if add_candidate_indices:
      n_vertices = len(output.tokens)

      graph = igraph.Graph(n=n_vertices + 1,
                           edges=[(position, position + length) for (position, length) in zip(positions, lengths)],
                           directed=True)
      boundaries = []
      s, t = 0, n_vertices
      for n in range(1, n_vertices):
        edges = []
        edges.extend(
          [(neighbor, n) for neighbor in graph.predecessors(n)])
        edges.extend(
          [(n, neighbor) for neighbor in graph.successors(n)])
        graph.delete_edges(edges)
        path_length = graph.shortest_paths(s, t)[0][0]
        if path_length == float("inf"):
          boundaries.append(n)
        graph.add_edges(edges)

      boundaries.sort()
      boundaries = boundaries + [len(tokens)]
      cand_indices = [[] for _ in boundaries]
      for index, (position, length) in enumerate(zip(positions, lengths)):
        right_index = bisect.bisect_left(boundaries, position + length)
        cand_indices[right_index].append(index)

    else:
      cand_indices = None

    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)

  def build_cls_encoding(self, add_candidate_indices=False):
    tokens = [self.cls_token]
    positions = [0]
    lengths = [1]
    cand_indices = [] if add_candidate_indices else None
    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)

  def build_sep_encoding(self, add_candidate_indices=False):
    tokens = [self.sep_token]
    positions = [0]
    lengths = [1]
    cand_indices = [] if add_candidate_indices else None
    return LatticeEncoding(tokens=tokens, lengths=lengths,
                           positions=positions, cand_indices=cand_indices)

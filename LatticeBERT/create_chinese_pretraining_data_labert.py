# coding=utf-8
# Copyright 2021 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tokenization
import tokenization_labert
from tokenization_labert import LatticeEncoding
import tensorflow as tf
import numpy as np
from create_pretraining_data_utils import (
  write_lattice_instances_to_example_files,
  write_lattice_instance_to_example_file,
  TrainingInstance,
  MaskedLmInstance)

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "input_file", None,
  "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
  "output_file", None,
  "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
  "vocab_file", None,
  "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
  "lexicon_file", None,
  "The lexicon file.")

flags.DEFINE_boolean(
  "use_named_lexicon", False,
  "The lexicon file is named (say in the format of {entry}\t{name}).")

flags.DEFINE_bool(
  "do_lower_case", True,
  "Whether to lower case the input text. Should be True for uncased "
  "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("next_sentence_type", 3, "Next sentence prediction: 2 or 3")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
  "dupe_factor", 10,
  "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
  "short_seq_prob", 0.1,
  "Probability of creating sequences which are shorter than the "
  "maximum length.")

flags.DEFINE_integer("max_position", 2040, "max_value of position")


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, next_sentence_type, rng,):
  """Create `TrainingInstance`s from raw text."""
  assert next_sentence_type in (2, 3), "next_sentence_type support 2 or 3 only"
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])

        lattice_encoding = tokenization_labert.LatticeEncoding()
        for span in line.split():
          new_lattice_encoding = tokenizer.tokenize(span, add_candidate_indices=True)
          lattice_encoding.extend(new_lattice_encoding)

        if len(lattice_encoding.tokens) > 0:
          all_documents[-1].append(lattice_encoding)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  tf.compat.v1.logging.info(f'Finished load {len(all_documents)} documents.')

  vocab_words = list(tokenizer.vocab.keys())
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      for instance in create_instances_from_document(all_documents, document_index,
                                                     max_seq_length, short_seq_prob,
                                                     masked_lm_prob, max_predictions_per_seq, next_sentence_type,
                                                     vocab_words, rng, tokenizer):
        yield instance


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, next_sentence_type, vocab_words, rng, tokenizer):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    current_chunk.append(document[i])

    current_length += len(document[i].tokens)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        encoding_a = LatticeEncoding()
        for j in range(a_end):
          encoding_a.extend(current_chunk[j])

        encoding_b = LatticeEncoding()
        # Random next
        if len(current_chunk) == 1 or rng.random() < (1. / next_sentence_type):
          # segment_b is from another document.
          next_sentence_label = 0
          target_b_length = target_seq_length - len(encoding_a.tokens)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            random_segment_tokens = random_document[j]
            encoding_b.extend(random_segment_tokens)
            if len(encoding_b.tokens) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          for j in range(a_end, len(current_chunk)):
            encoding_b.extend(current_chunk[j])

          if next_sentence_type == 2:
            next_sentence_label = 1
          else:
            if rng.random() < 0.5:
              next_sentence_label = 1
            else:
              encoding_a, encoding_b = encoding_b, encoding_a
              next_sentence_label = 2

        truncate_seq_pair(encoding_a, encoding_b, max_num_tokens, rng)
        assert len(encoding_a.tokens) >= 1
        assert len(encoding_b.tokens) >= 1

        # [CLS] tokens_a [SEP] tokens_b [SEP]
        encodings = tokenizer.build_cls_encoding(add_candidate_indices=True)
        segment_ids = [0]

        encodings.extend(encoding_a)
        segment_ids.extend([0] * len(encoding_a.tokens))

        encodings.extend(tokenizer.build_sep_encoding(add_candidate_indices=True))
        segment_ids.append(0)

        encodings.extend(encoding_b)
        segment_ids.extend([1] * len(encoding_b.tokens))

        encodings.extend(tokenizer.build_sep_encoding(add_candidate_indices=True))
        segment_ids.append(1)

        (encodings, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
          encodings, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

        instance = TrainingInstance(
          encodings=encodings,
          segment_ids=segment_ids,
          next_sentence_label=next_sentence_label,
          masked_lm_positions=masked_lm_positions,
          masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


def create_masked_lm_predictions(encodings, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = encodings.cand_indices

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(encodings.tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for _ in cand_indexes:
    span_length = 1
    start_position = np.random.choice(len(cand_indexes), size=1)[0]

    length = 0
    for position in range(start_position, start_position + span_length):
      index_set = cand_indexes[position]
      length += len(index_set)

    if len(masked_lms) + length > num_to_predict:
      continue

    is_any_index_covered = False
    for position in range(start_position, start_position + span_length):
      index_set = cand_indexes[position]
      for index in index_set:
        if index in covered_indexes:
          is_any_index_covered = True
          break
      if is_any_index_covered:
        break
    if is_any_index_covered:
      continue

    for position in range(start_position, start_position + span_length):
      index_set = cand_indexes[position]
      for index in index_set:
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
          masked_token = "[MASK]"
        else:
          # 10% of the time, keep original
          if rng.random() < 0.5:
            masked_token = encodings.tokens[index]
          # 10% of the time, replace with random word
          else:
            masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        masked_lms.append(MaskedLmInstance(index=index, label=encodings.tokens[index]))
        encodings.tokens[index] = masked_token
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return encodings, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(encoding_a: LatticeEncoding,
                      encoding_b: LatticeEncoding, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    encoding_a_lazy_length = encoding_a.lazy_length()
    encoding_b_lazy_length = encoding_b.lazy_length()

    total_length = encoding_a_lazy_length + encoding_b_lazy_length
    if total_length <= max_num_tokens:
      break

    if encoding_a_lazy_length > encoding_b_lazy_length:
      trunc_tokens = encoding_a
      trunc_tokens_lazy_length = encoding_a_lazy_length
    else:
      trunc_tokens = encoding_b
      trunc_tokens_lazy_length = encoding_b_lazy_length

    assert trunc_tokens_lazy_length >= 1

    if rng.random() < 0.5:
      trunc_tokens.lazy_pop_front()
    else:
      trunc_tokens.lazy_pop_back()

  encoding_a.finalize_lazy_pop_front()
  encoding_a.finalize_lazy_pop_back()
  encoding_b.finalize_lazy_pop_front()
  encoding_b.finalize_lazy_pop_back()


def main_func(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if FLAGS.use_named_lexicon:
    tokenizer = tokenization_labert.LatticeTokenizerWithMapping(
      vocab_file=FLAGS.vocab_file,
      lexicon_file=FLAGS.lexicon_file,
      do_lower_case=FLAGS.do_lower_case)
  else:
    tokenizer = tokenization_labert.LatticeTokenizer(
      vocab_file=FLAGS.vocab_file,
      lexicon_file=FLAGS.lexicon_file,
      do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  tf.compat.v1.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.compat.v1.logging.info("  %s", input_file)

  np.random.seed(FLAGS.random_seed)
  rng = random.Random(FLAGS.random_seed)

  total_written = 0
  tf.compat.v1.logging.info(f"*** Writing to output {FLAGS.output_file} ***")
  writer = tf.io.TFRecordWriter(FLAGS.output_file)
  for inst_index, instance in enumerate(
      create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        FLAGS.next_sentence_type, rng,)):

    write_lattice_instance_to_example_file(
      instance, tokenizer, writer,
      FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,
      position_embedding_names=('start', 'end'),
      do_dump_example=inst_index < 20)

    total_written += 1

  writer.close()

  tf.compat.v1.logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("lexicon_file")
  tf.compat.v1.app.run(main=main_func)

# coding=utf-8
# Copyright 2021 Alibaba DAMO NLP Team Authors.
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

import os
import tempfile
import tokenization_labert
import six
import tensorflow as tf


class TokenizationLaBertTest(tf.test.TestCase):

  def test_full_tokenizer(self):
    vocab_tokens = [
      "[UNK]", "[CLS]", "[SEP]", "州", "市", "是", "中", "华",
      "人", "民", "共", "和", "国",
      "安", "徽", "省", "下", "辖",
      "的", "地", "级", "市", "。"
    ]
    lexicon_tokens = [
      "州市", "中华", "人民", "华人", "共和", "共和国",
      "中华人民共和国", "国安", "安徽", "安徽省", "下辖",
      "地级市",
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      if six.PY2:
        vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
      else:
        vocab_writer.write("".join(
            [x + "\n" for x in vocab_tokens]).encode("utf-8"))

      vocab_file = vocab_writer.name

    with tempfile.NamedTemporaryFile(delete=False) as lexicon_writer:
      if six.PY2:
        lexicon_writer.write("".join([x + "\n" for x in lexicon_tokens]))
      else:
        lexicon_writer.write("".join(
            [x + "\n" for x in lexicon_tokens]).encode("utf-8"))

      lexicon_file = lexicon_writer.name

    tokenizer = tokenization_labert.LatticeTokenizer(vocab_file, lexicon_file, do_lower_case=True)
    os.unlink(vocab_file)
    os.unlink(lexicon_file)

    encodings = tokenizer.tokenize("州市是中华人民共和国安徽省下辖的地级市。")
    print(encodings)
    #cls_token = tokenizer.build_cls_token()
    #self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

    #self.assertAllEqual(
    #    tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

  def test_full_tokenizer2(self):

    tokenizer = tokenization_labert.LatticeTokenizer(
      './data/bert_chinese_vocab.txt', './data/lexicon.txt', do_lower_case=True)

    encodings = tokenizer.tokenize("基礎數學的知識與運用總是個人與團體生活中不可或缺的一環。")
    print(encodings)
    #cls_token = tokenizer.build_cls_token()
    #self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

    #self.assertAllEqual(
    #    tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])


if __name__ == "__main__":
  tf.test.main()

LatticeBERT
===========

## Introduction

We propose a novel pre-training paradigm for Chinese — `LatticeBERT` 
which explicitly incorporates word representations with those of characters, 
thus can model a sentence in a multi-granularity manner.

Paper: [Lattice-BERT: Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models](url-to-be-updated)


## How we create multi-granularity inputs.

We create multi-granularity inputs as:

- We create a BertTokenizer using character vocabulary.
- We parse the lexicon with the BertTokenizer to get the sequence of characters for each word.
- We test if a span of characters matches the sequence. If matched, we treated it as a word. 
  We use Trie for efficient span search. 
- The size of vocabulary can be computed as the size of the character vocabulary
  plus the size of word vocabulary. 

### Lexicon entry with term cluster in mind

In the real world case, we might have created
a lexicon with many entries.
Too many lexicon entries will and consume too much memory footprint
and slow down the model.
Meanwhile, we observe that some long-tailed words
can be clustered.
To support the lexicon with word-cluster mappings,
we create a tokenizer that creates multi-granularity inputs of term cluster.
The lexicon is in the format of `{surface_form}\t{cluster_form}`.
Please refer the `tokenization_labert.py` for implementation details.


## Pre-trained models

### Downloads

We provide three models of different sizes,
including

- `tiny` with `E=128, H=256, L=4, A=4`: [[Downloads]](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/LatticeBERT/chinese_labert-tiny-std-512.tar.gz)
- `lite` with `E=128, H=512, L=6, A=8`: [[Downloads]](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/LatticeBERT/chinese_labert-lite-std-512.tar.gz)
- `base` with `E=128, H=768, L=12, A=12`: [[Downloads]](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/LatticeBERT/chinese_labert-base-std-512.tar.gz)

where `E` is the dimension of embedding, `H` is the dimension of hidden states,
and `L` is the number of layers, and `A` is the number of heads in the self-attention.
We set the size of post-attention feed-forward as `4*H`.

The file structure of the unzipped directory is
```
tree
.
├── checkpoint
├── labert_config.json
├── lexicon.txt
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
└── vocab.txt
```

### Lexicon in the pre-trained models

Our word vocabulary is made of 102K high-frequency open domain words.

## Fine-tuning with LatticeBERT

### Prepare your data

Please put your training, development, and test data in the same folder,
the structure is
```
tree data
data
├── dev.txt
├── test.txt
└── train.txt
```

#### Data format for sentence classification 
For a single sentence classification, the format is `{sentence1}\t{label}` line by line.
For a pair sentence classification, the format is `{sentence1}\t{sentence2}\t{label}` line by line.

#### Data format for sequence labeling 
We set the format of sequence labeling input as the CoNLL-03 format
```
{sentence1_token1}\t{label1}
{sentence1_token2}\t{label2}

{sentence2_token1}\t{label1}
{sentence2_token2}\t{label2}
```

### AFQMC as an example
We use the `AFQMC` dataset as a fine-tuning example.
First download the data from https://github.com/CLUEbenchmark/CLUE.
Then convert the data into the formatted file: `{sentence1}\t{sentence2}\t{label}`
with the following code.

```python
import json
import sys
with open(sys.argv[1], 'r') as reader:
  for line in reader:
    data = json.loads(line)
    print(f'{data["sentence1"]}\t{data["sentence2"]}\t{data["label"]}')
```

To fine-tuning a AFQMC classification model, please run the following commands.
```shell
export LABERT_DIR=/path/to/your/labert_ckpt/

CUDA_VISIBLE_DEVICES="1" python run_classifier_labert.py \
--init_checkpoint=${LABERT_DIR} \
--data_dir=/path/to/your/afqmc/data \
--labert_config_file=${LABERT_DIR}/labert_config.json \
--lexicon_file=${LABERT_DIR}/lexicon.txt \
--vocab_file=${LABERT_DIR}/vocab.txt \
--task_name=pair \
--use_named_lexicon=true \
--do_train=true \
--do_eval=true \
--learning_rate=5e-5 \
--num_train_epochs=30 \
--output_dir=/tmp/afqmc
```

If you use the `Chinese-LaStructBERT-tiny` as the initial checkpoint,
you can get a development accuracy of about `70`.

- **[note]** The meaning of the parameters is basically the same as those google bert.
- **[note]** LatticeBERT generally requires more training epochs to reach good performance.

## Pre-training with LatticeBERT

When applying LatticeBERT to some specific-domain,
we may expect to use our own vocabulary, rather than the open-domain one
shipped with the off-shell model.
We support pre-training a LatticeBERT with domain-specific vocabulary.
Roughly speaking, it can be achieved with the following steps:

- Prepare your own lexicon
- Port the open-domain LatticeBERT to the domain-specific LatticeBERT
  by re-using as many parameters as possible. This step will create
  a domain-specific LatticeBERT checkpoint.
- Run the pre-training with the new checkpoint.

### Prepare your own lexicon

### Port the open-domain LatticeBERT to specific domain

```shell
python port_labert_checkpoint.py \
--init_checkpoint=/path/to/checkpoint \
--lexicon=/path/to/target/lexicon.txt \
--output_dir=/path/to/new/checkpoint
```

### Create tfrecords for pre-training

```shell
python create_chinese_pretraining_data_labert.py \
--use_named_lexicon=true \
--vocab_file=/path/to/new/checkpoint/vocab.txt \
--input_file=data/sample.doc \
--output_file=data/sample.tfrecord \
--lexicon_file=/path/to/new/checkpoint/lexicon.txt
```

The other parameters have the same meaning as `google-bert`.

### Run pre-training

```shell
python run_pretraining_labert.py \
--init_checkpoint=/path/to/new/checkpoint/ \
--input_file=data/sample.tfrecord \
--eval_file=data/sample.tfrecord \
--labert_config_file=/path/to/new/checkpoint/labert_config.json \
--output_dir=/tmp/labert_pretrain \
--do_train
```

- **[note]** For more information about parameters, please refer `python run_pretraining_labert.py --help`
- **[note]** We use horovod for multi-gpu training. Set `--use_horovod` and run
  with `horovodrun -np ${NUM_PROCESS} python run_pretrianing_labert.py ...`
# SOFA (Supporter Of Foundation Ai)

***Alice lies on the SOFA, opens her Mind.***


SOFA aims to faciliate easy use and distribution of the pretrained language models from Alibaba DAMO Academy AliceMind project. 
In addition, detail examples in the project make it simple for any end-user to access those models.

SOFA has the following features:

# Features

- **AliceMind: ALIbaba's Collection of Encoder-decoders from MinD (Machine IntelligeNce of Damo) Lab** in the form of model scheme from [transformers](https://github.com/huggingface/transformers), including:
    * Large-scale language model: PLUG, a chinese pre-training model for language understanding and generation. PLUG has 27 billion parameters. The training of PLUG is two-stage, the first stage is a 24-layer [StructBERT](https://arxiv.org/abs/1908.04577) encoder, and the second stage is a 24-6-layer [PALM](https://arxiv.org/pdf/2004.07159.pdf?fbclid=IwAR0BNl1IzR5bhcuEbyfNw2UN7MApHFoFP3BN40FKkW8x3bqolK_HilU293I) encoder-decoder.
    * Language understanding model: [StructBERT](https://arxiv.org/abs/1908.04577) (```ICLR 2020```)， a model extended from BERT, by incorporating language structures into pre-training. Specifically, we pre-train StructBERT with two auxiliary tasks to make the most of the sequential order of words and sentences, which leverage language structures at the word and sentence levels, respectively. On the other handby applying the ***adversarial training*** during the fine-tune, the StructBERT with size large has **4~8%** better performace than that in BERT 
    * Generative language model: [PALM](https://arxiv.org/abs/2004.07159) (```EMNLP 2020```),  a novel scheme that jointly pre-trains an autoencoding and autoregressive language model on a large unlabeled corpus, specifically designed for generating new text conditioned on context.
    * Cross-lingual language model: [VECO](https://arxiv.org/abs/2010.16046) (```ACL 2021```), a variable encoder-decoder (VECO) model targets at providing pre-trained model initialization for both the encoder-only and encoder-decoder Transformer with the most streamlined parameters
    * more models are coming soon!

-  **Compatible with open-source library like ***transformers*****, user can use all of the models above in their original ***transformer*** environment as well. 
    * By the design of the replaceable run-time trainer, SOFA can easily extend to other open-source library other than ***transformers***, such as [EasyNLP](https://github.com/alibaba/EasyNLP)
    * Specifically, after setting ***transformers*** as runtime backend, user can inject the Model/Tokenizer/Config/Pipeline into the ***transformers***'s ***MAPs***，then, by calling the `AutoModel.from_pretrained('SturctBert')`, SOFA automatically inference the model as StructBert.
    * At last, SOFA can run independently without import any open-source library like ***transformers***, SOFA has ***AutoModel\Trainer\Optimizer*** component as well, in addition, our ***AutoModel*** allow user use the http link of zipped model as input.

- **Integrated effective fine-tuning techniques**, user can trial on those cutting-edge techniques to get better performances during fine-tuning:
    * ***CHILD-TUNING*** updates a subset of parameters (called child network) of large pretrained models via strategically masking out the gradients of the non-child network during the backward process. By applying `sofa.apply_child_tuning_to_trainer`, user can try the method in SOFA, or even with user-defined optimizer.
    * more techs are coming soon!
    
# Prerequisite

python >= 3.6.
transformers >= 4.10.0

# Installation

Setup from source

```shell
git clone https://github.com/alibaba/AliceMind.git
cd AliceMind/sofa
python3 setup.py install
```

# Quick Start


Once you have installed the SOFA package, and have had **transformers** and **datasets** in your environment, following steps give you a quick example how to run StructBERT from AliceMind quickly.

Firstly, downloading the model files of StructBERT-large on English tasks to the location of `/tmp/english_sbert-large-std-512`

```shell
wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/english_sbert-large-std-512.tar.gz -P /tmp
mkdir -p /tmp/english_sbert-large-std-512
tar xvf /tmp/english_sbert-large-std-512.tar.gz -C /tmp/english_sbert-large-std-512
```

Then, importing SOFA and models. 
The `sofa.environ("huggingface")` does all the tricks that inject or register AliceMind models to the transformers' AutoModel.

```python
import sofa
sofa.environ("huggingface")
```


Now, loading datasets. 
In this case, the dataset `imdb` from [transformers](https://huggingface.co/docs/transformers/training) is used：
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

Loading tokenizer from the downloaded model files and tokenizing the datasets,

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/tmp/english_sbert-large-std-512", model_max_length=128)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print(tokenizer.__class__)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
 
# pruned from the original datasets, in order to run test case faster
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

```
Loading StructBERT model from huggingface's auto model method.

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("/tmp/english_sbert-large-std-512", num_labels=2)
print(model)
```
or loading StructBERT explicitly
```python
from sofa import SbertTokenizerFast, SbertForSequenceClassification
tokenizer = SbertTokenizerFast.from_pretrained("/tmp/english_sbert-large-std-512", model_max_length=128)
model = SbertForSequenceClassification.from_pretrained("/tmp/english_sbert-large-std-512", num_labels=2)
print(tokenizer.__class__)
print(model)
```


Now training the model with the Child-Tuning method.
```python
from transformers import TrainingArguments, Trainer
from sofa import apply_child_tuning_to_trainer

training_args = TrainingArguments(output_dir="test_trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)
apply_child_tuning_to_trainer(trainer)
trainer.train()
```

To learn more, please refer [Tutorial](./Tutorial.md) 


# AliceMind Models

## StructBert

StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding

[https://arxiv.org/abs/1908.04577](https://arxiv.org/abs/1908.04577)

#### Introduction

Please check the model's [readme file](../StructBERT/README.md).

#### Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|structbert.en.large | StructBERT using the BERT-large architecture | 340M | [structbert.en.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/english_sbert-large-std-512.tar.gz) |
|structbert.ch.large | Chinese StructBERT; BERT-large architecture | 330M | [structbert.ch.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/chinese_sbert-large-std-512.tar.gz) |

#### Results
The results of GLUE & CLUE tasks can be reproduced using the hyperparameters listed in the following "Example usage" section.
##### structbert.en.large
[GLUE benchmark](https://gluebenchmark.com/leaderboard)

|Model| MNLI | QNLIv2 | QQP | SST-2 |
|--------------------|-------|-------|-------|-------|
|structbert.en.large |86.62% |93.34% |91.99% |93.00% |
##### structbert.ch.large
[CLUE benchmark](https://www.cluebenchmarks.com/)

|Model | TNEWS | CSL | CMNLI | CLUEWSC | IFLYTEK | OCNLI | AFQMC | CLUENER(F1) |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|structbert.ch.large |61.34% |83.43% |84.09% |90.79% |62.79% | 80.70% | 77.32% | 77.98% |



## Veco

VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation

[Paper Link](https://arxiv.org/abs/2010.16046)

#### News
- May, 2021: [VECO](https://arxiv.org/abs/2010.16046) was accepted by ACL 2021.
- Mar, 2021: VECO ranks first at the [XTREME](https://sites.research.google/xtreme/) leaderboard.

#### Introduction

Please check the model's [readme file](../VECO/README.md).

#### Pre-trained models

##### Pre-trained models for Multilingual NLU tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder layers trained on 50 languages' monolingual and bilingual corpus | 550M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLU/model/xtreme-released-veco.tar.gz)


##### Pre-trained models for Multilingual NLG tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder and decoder layers trained on 50 languages' monolingual and bilingual corpus | 660M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/model/veco-large.tar.gz)


#### Results

##### Results of Multilingual NLU tasks

[XTREME](https://sites.research.google/xtreme/) is one of the most representative massively multilingual benchmark.

The result of XNLI task is reproduced using the default hyperparameters listed in `finetune_veco.sh` file. Other benchmark on tasks such as XQuAD, Tatoeba can be found in [VECO-Readme](https://github.com/alibaba/AliceMind/tree/main/VECO)

|Model| XNLI<br>(Acc) |
|--------------------|-------|
|veco.large | 79.63 |


### PALM

PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation

[Paper Link](https://arxiv.org/abs/2004.07159)

#### Introduction

Please check the model's [readme file](../PALM/README.md).

#### Pre-trained models

|Model | Description | #params | Download |
---|---|---|---
palm.ch.base | PALM using a 12-layer encoder and a 12-layer decoder on Chinese training data | 217M | [chinese-palm-base.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/chinese-palm-base.tar.gz)

#### Results

##### palm.ch.base
[DuReader benchmark](https://github.com/baidu/DuReader)

[CLGE benchmark](https://github.com/CLUEbenchmark/CLGE)

|palm.ch.base | DuReaderQG | DuReader-Robust | LCSTS |
|--------------------|-------|-------|-------|
|BLEU-1 | 0.5863 | 0.5560 | 0.3296 |
|BLEU-4 | 0.3982 | 0.3553 | 0.1716 |
|ROUGE-L | 0.6027 | 0.5661 | 0.3293 |


## PLUG

PLUG, a large-scale chinese pre-training model for language understanding and generation.


#### Introduction

Please check the model's [readme file](../PLUG/README.md).

#### Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|PLUG |chinese PLUG  | 27B | [PLUG](https://github.com/alibaba/AliceMind/tree/main/PLUG#pre-trained-model-download) |

#### Results

##### Fine-tuning on Question Generation
The result of Question Generation task is reproduced using the default hyperparameters listed in `finetune_plug.sh` file. Other benchmark on datasets such as [KBQG](https://github.com/nanduan/NLPCC-KBQA), [DuReaderQG](https://arxiv.org/abs/1711.05073) can be found in [PLUG-Readme](https://github.com/alibaba/AliceMind/tree/main/PLUG)
|PLUG| BLEU-1|BLEU-2|BLEU-4|ROUGE-L|
|----|-------|------|------|-------|
|DuReader-Robust|0.6310|0.5510|0.4223|0.6264|

*Device Requirements for finetune: single node, 8 32G V100.

#### 免责声明
针对基于本模型由用户直接或间接生成的内容，我们不对基于模型产生、生成内容的合规性承担责任。

您在此承诺：

1、您会严格落实法律法规、部门规章等相关文件的全部要求（包括但不限于备案、标识等），不会使用模型能力从事任何违反法律法规或公序良俗的行为，不会使用模型能力生成任何涉及恐怖、暴力、色情等违反法律要求的信息，或为上述行为提供帮助。

2、您使用模型能力而取得的任何产出或成果（包括但不限于音频、视频、文本文件等），应按照法律法规、部门规章等相关文件的要求进行合法使用。

3、您承诺您在使用模型能力过程中提供的素材等数据，是您通过合法途径取得并获得充分授权对其进行使用的。您承诺使用的任何素材、相关数据、对模型能力的使用及使用能力所产生的任何成果，均未侵犯任何第三方的合法权益。如有第三方基于侵犯著作权、侵犯第三人之权益或违反中国法律法规或其他适用的法律等原因而向提起索赔、诉讼或可能向提起诉讼，则您应赔偿因此承担的所有费用或所有损失。



# AliceMind Techniques
### Child-Tuning

Child-Tuning is a new and effective fine-tuning technique which can be used in various optimizers. 
The technique updates a subset of parameters (called child network) of large pretrained models via strategically 
masking out the gradients of the non-child network during the backward process, user can simply treat it as 
a drop-out in backward-propagation.
The technique can achieve an **_1.5 ~ 8.6 percent of accuracy improvement_**, and is _**easy to import to your code**_.

> Experiments on various downstream tasks in GLUE benchmark show that Child-Tuning consistently 
> outperforms the vanilla fine-tuning by 1.5~8.6 average score among four different pretrained models, 
> and surpasses the prior fine-tuning techniques by 0.6~1.3 points. Furthermore, 
> empirical results on domain transfer and task transfer show that Child-Tuning can obtain 
> better generalization performance by large margins.

The article can be viewed [here](https://arxiv.org/abs/2109.05687).

We strongly suggest user to try this technique, it's effective on all most all the datasets, especially small datasets.
Here is the performances in the single classification tasks:

|  Dataset   | Accuracy with child-tuning-D  | Accuracy with child-tuning-F  | Accuracy without any child-tuning |
| :----: | :----: | :----: | :----: |
|  cluewsc | 86.8%  | 87.5%  | 86.18%  |



# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/AliceMind/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. 

# Others
Other than SOFA, please check [AliceMind Open Platform](https://alicemind.aliyuncs.com/#/home) for more infomration.

# Contact Us
Scan the following QR codes to join Dingtalk discussion group. The group discussions are mostly in Chinese, but English is also welcomed.

<img src="https://cdn.nlark.com/yuque/0/2022/png/2963703/1650986085790-38964ecc-5779-4b51-b4b5-367110e51824.png?x-oss-process=image%2Fresize%2Cw_878%2Climit_0" width="300"/>













# VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation
[Paper Link](https://arxiv.org/abs/2010.16046)

## News
- May, 2021: [VECO](https://arxiv.org/abs/2010.16046) was accepted by ACL 2021.
- Mar, 2021: VECO ranks first at the [XTREME](https://sites.research.google/xtreme/) leaderboard.

## Introduction

VECO is a variable encoder-decoder (VECO) model targets at providing pre-trained model initialization for both the encoder-only and
encoder-decoder Transformer with the most streamlined parameters. . As a result, VECO delivers new state-of-the-art
results on various cross-lingual understanding tasks of the XTREME benchmark
covering text classification, sequence labeling, question answering, and sentence
retrieval. For generation tasks, VECO also outperforms all existing cross-lingual
models and state-of-the-art Transformer variants on WMT14 English-to-German
and English-to-French translation datasets, with gains of up to 1∼2 BLEU.


## Pre-trained models

### Pre-trained models for Multilingual NLU tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder layers trained on 50 languages' monolingual and bilingual corpus | 550M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLU/model/xtreme-released-veco.tar.gz)


### Pre-trained models for Multilingual NLG tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder and decoder layers trained on 50 languages' monolingual and bilingual corpus | 660M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/model/veco-large.tar.gz)


## Results

### Results of Multilingual NLU tasks 

[XTREME](https://sites.research.google/xtreme/) is one of the most representative massively multilingual benchmark.

The results of XNLI, XQuAD and Tatoeba task can be reproduced using the default hyperparameters listed in `.sh` file.

|Model| XNLI<br>(Acc) | XQuAD<br>(F1/EM) | Tatoeba<br>(Acc) |
|--------------------|-------|-------|-------|
|veco.large | 79.9 | 77.5/61.9 | 75.1 |

### Results of Multilingual NLG tasks 

Results on custom machine translation datasets.

Model |  WMT14 En-Fr<br>tok/detok-BLEU | WMT14 En-De<br>tok/detok-BLEU
---|---|---
`XLM-R (24 encoder + 6 decoder)` | 43.7/41.1 | 30.8/29.9
`mBART (12 encoder + 12 decoder)` | 43.2/41.0 | 30.0/29.1
`VECO (24 encoder + 6 decoder)` | 44.4/42.0 | 31.5/30.5 


## Finetuning

- [Finetuning on multilingual NLU tasks in the xtreme benchmark](NLU/README.md)
- [Finetuning on cross-lingual NLG tasks (e.g., machine translation)](NLG/README.md)

## Citation

```bibtex
@article{Luo2020VECO
  author    = {Fuli Luo and Wei Wang and Jiahao Liu and Yijia Liu and Bin Bi and Songfang Huang and Fei Huang and Luo Si},
  title     = {{VECO:} Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation},
  journal   = {CoRR},
  volume    = {abs/2010.16046},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.16046},
  archivePrefix = {arXiv},
  eprint    = {2010.16046},
}
```

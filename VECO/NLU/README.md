# NLU Task Fine-tune On VECO

[**Download**](#download-the-data) |
[**Fine-tune**](#run-fine-tune) |
[**Result**](#result) |

This repository contains fine-tune code of NLU tasks on VECO, and the code base is built on top of [XTREME](https://github.com/google-research/xtreme).

# Pre-trained models

Model | Description | # Params | Download
---|---|---|---
`VECO_large` | VECO model with 24 encoder layers trained on 50 languages' monolingual and bilingual corpus | 550M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLU/model/xtreme-released-veco.tar.gz)


# Download the data

In order to run this code, the first step is to download the dependencies. We assume you have installed [`anaconda`](https://www.anaconda.com/) and use Python 3.7+. The additional requirements including `transformers`, `seqeval` (for sequence labelling evaluation), `tensorboardx`, `jieba`, `kytea`, and `pythainlp` (for text segmentation in Chinese, Japanese, and Thai), and `sacremoses` can be installed by running the following script:
```
bash install_tools.sh
```

The next step is to download the data. To this end, first create a `download` folder with ```mkdir -p download``` in the root of this project, and then run the following command to download the datasets:
```
bash scripts/download_data.sh
```

# Run fine-tune

The evaluation setting in XTREME is zero-shot cross-lingual transfer from English. We fine-tune VECO models on the labelled data of XNLI and XQuAD task in English. Each fine-tuned model is then applied to the test data of the same task in other languages to obtain predictions.

## Sentence Classification Task (XNLI)

The second sentence classification dataset is the Cross-lingual Natural Language Inference (XNLI) dataset. You can fine-tune a pre-trained multilingual model on the English MNLI data with the following command:
```
bash scripts/train_xnli.sh [MODEL_PATH] [GPU] [LR] [EPOCH] [BATCH_SIZE]
```

## Question Answering Task (XQuAD)

For XQuAD, the model should be trained on the English SQuAD training set. Using the following command, you can first fine-tune a pre-trained multilingual model on SQuAD English training data, and then you can obtain predictions on the test data of XQuAD task.
```
bash scripts/train_squad.sh [MODEL_PATH] [GPU] [LR] [EPOCH] [BATCH_SIZE]
```

## Sentence Retrieval Task (Tatoeba)

The second cross-lingual sentence retrieval dataset we use is the Tatoeba dataset. You can directly apply the model to obtain predictions on the test data of the task:
```
bash scripts/run_tatoeba.sh [MODEL_PATH] [GPU] [LAYER]
```

# Result

The results of XNLI, XQuAD and Tatoeba task can be reproduced using the default hyperparameters listed in `.sh` file.

|Model| XNLI<br>(Acc) | XQuAD<br>(F1/EM) | Tatoeba<br>(Acc) |
|--------------------|-------|-------|-------|
|`VECO_large` | 79.9 | 77.5/61.9 | 75.1 |

# Paper

If you use this benchmark or the code in this repo, please cite the following papers `\cite{Luo2020VECO}` and `\cite{hu2020xtreme}`.
```
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

@article{hu2020xtreme,
      author    = {Junjie Hu and Sebastian Ruder and Aditya Siddhant and Graham Neubig and Orhan Firat and Melvin Johnson},
      title     = {XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}
```

Please consider including a note similar to the one below to make sure to cite all the individual datasets in your paper.

```
@inproceedings{Conneau2018xnli,
    title = "{XNLI}: Evaluating Cross-lingual Sentence Representations",
    author = "Conneau, Alexis  and
      Rinott, Ruty  and
      Lample, Guillaume  and
      Williams, Adina  and
      Bowman, Samuel  and
      Schwenk, Holger  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of EMNLP 2018",
    year = "2018",
    pages = "2475--2485",
}

@inproceedings{artetxe2020cross,
author = {Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
booktitle = {Proceedings of ACL 2020},
title = {{On the Cross-lingual Transferability of Monolingual Representations}},
year = {2020}
}

@article{Artetxe2019massively,
author = {Artetxe, Mikel and Schwenk, Holger},
journal = {Transactions of the ACL 2019},
title = {{Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond}},
year = {2019}
}
```

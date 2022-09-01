# PST
Source code for IJCAI 2022 Long paper: Parameter-Efficient Sparsity for Large Language Models Fine-Tuning.


## 🔥 Introduction

With the dramatically increased number of parameters in language models, sparsity methods have received ever-increasing research focus to compress and accelerate the models. While most research focuses on how to accurately retain appropriate weights while maintaining the performance of the compressed model, there are challenges in the computational overhead and memory footprint of sparse training when compressing large-scale language models. To address this problem, we propose a Parameter-efficient Sparse Training (PST) method to reduce the number of trainable parameters during sparse-aware training in downstream tasks. Specifically, we first combine the data-free and data-driven criteria to efficiently and accurately measure the importance of weights. Then we investigate the intrinsic redundancy of data-driven weight importance and derive two obvious characteristics i.e. low-rankness and structuredness. Based on that, two groups of small matrices are introduced to compute the data-driven importance of weights, instead of using the original large importance score matrix, which therefore makes the sparse training resource-efficient and parameter-efficient.

You can refer to our [paper](https://arxiv.org/abs/2205.11005) for more details.

## 🏋🏻‍♂️ Repository Overview

There are several directories in this repo:

pst/ contains the source code for the package pst;

NLG/ contains an example implementation of PST in GPT-2 using our package, which can be used to reproduce the result in our paper;

NLU/ contains an example implementation of PST in BERT and RoBERTa using our package, which can be used to reproduce the result in our paper;

## 🚀 Quickstart

1. Install the dependencies
```bash
pip3 install -r requirement.txt
```

2. Import PST library
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

3. Training network in [NLU](./NLU) and [NLG](./NLG)


## 🌝 Citation

If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{li-etal-2022-pst,
    title = "Parameter-Efficient Sparsity for Large Language Models Fine-Tuning",
    author = "Yuchao Li and Fuli Luo and Chuanqi Tan and Mengdi Wang and Songfang Huang and Shen Li and Junjie Bai",
    booktitle = "31th International Joint Conference on Artificial Intelligence",
    year = "2022"
}
```

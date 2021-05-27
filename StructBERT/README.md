# StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding

[https://arxiv.org/abs/1908.04577](https://arxiv.org/abs/1908.04577)

## Introduction
We extend BERT to a new model, StructBERT, by incorporating language structures into pre-training. 
Specifically, we pre-train StructBERT with two auxiliary tasks to make the most of the sequential 
order of words and sentences, which leverage language structures at the word and sentence levels, 
respectively.
## Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|structbert.en.large | StructBERT using the BERT-large architecture | 340M | [structbert.en.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/en_model) |
|structroberta.en.large | StructRoBERTa continue training from RoBERTa | 355M | Coming soon |
|structbert.ch.large | Chinese StructBERT; BERT-large architecture | 330M | [structbert.ch.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model) |

## Results
The results of GLUE & CLUE tasks can be reproduced using the hyperparameters listed in the following "Example usage" section.
#### structbert.en.large
[GLUE benchmark](https://gluebenchmark.com/leaderboard)

|Model| MNLI | QNLIv2 | QQP | SST-2 | MRPC | 
|--------------------|-------|-------|-------|-------|-------|
|structbert.en.large |86.86% |93.04% |91.67% |93.23% |86.51% |
#### structbert.ch.large
[CLUE benchmark](https://www.cluebenchmarks.com/)

|Model | CMNLI | OCNLI | TNEWS | AFQMC |
|--------------------|-------|-------|-------|-------|
|structbert.ch.large |84.47% |81.28% |68.67% |76.11% | 

## Example usage
#### Requirements and Installation
* [PyTorch](https://pytorch.org/) version >= 1.0.1

* Install other libraries via
```
pip install -r requirements.txt
```

* For faster training install NVIDIA's [apex](https://github.com/NVIDIA/apex) library

#### Finetune MNLI

```
python run_classifier_multi_task.py \
  --task_name MNLI \
  --do_train \
  --do_eval \
  --do_test \
  --amp_type O1 \
  --lr_decay_factor 1 \
  --dropout 0.1 \
  --do_lower_case \
  --detach_index -1 \
  --core_encoder bert \
  --data_dir path_to_glue_data \
  --vocab_file config/vocab.txt \
  --bert_config_file config/large_bert_config.json \
  --init_checkpoint path_to_pretrained_model \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --fast_train \
  --gradient_accumulation_steps 1 \
  --output_dir path_to_output_dir 
```

## Citation
If you use our work, please cite:
```
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```

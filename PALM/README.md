# PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation

[https://arxiv.org/abs/2004.07159](https://arxiv.org/abs/2004.07159)

## Introduction
This work presents PALM with a novel scheme that jointly pre-trains an autoencoding and \
autoregressive language model on a large unlabeled corpus, specifically designed for \
generating new text conditioned on context.
## Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|palm.en.base | PALM using a 12-layer encoder and a 12-layer decoder | 257M | [palm model and cnndm data](https://drive.google.com/file/d/1mSp-4KfBwGKUAdWiW-ctOR9Qgi0a-w9B/view?usp=sharing) |
|palm.en.large | PALM using a 24-layer encoder and a 6-layer decoder | 483M | Coming soon |

## Example usage
#### Requirements and Installation
* [PyTorch](https://pytorch.org/) version == 1.1.0
* Install other libraries via
```
pip install -r requirements.txt
```
Some codes are borrowed from [PreSumm](https://github.com/nlpyang/PreSumm)
#### Finetune CNN/DailyMail
Download the processed data ([palm model and cnndm data](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/LatticeBERT/chinese_labert-base-std-512.tar.gz))
```
sh finetune_cnndm_task_roberta.sh 
```
#### Other Generation Task
Process your data
```
sh process_data.sh
```
## Citation
If you use our work, please cite:
```
@inproceedings{bi-etal-2020-palm,
    title = "{PALM}: Pre-training an Autoencoding{\&}Autoregressive Language Model for Context-conditioned Generation",
    author = "Bi, Bin and Li, Chenliang and Wu, Chen and Yan, Ming and
      Wang, Wei and Huang, Songfang and Huang, Fei and Si, Luo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.700",
}
```

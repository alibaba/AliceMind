# SDCUP: Schema Dependency Enhanced Curriculum Pre-Training for Table Semantic Parsing

## Introduction
We design a schema dependency pre-training objective to impose the desired inductive bias into the learned representations for table pre-training. We further propose a schema-aware curriculum learning approach to alleviate the impact of noise and learn effectively from the pre-training data in an easy-to-hard manner. The experiment results on SQUALL and Spider demonstrate the effectiveness of our pre-training objective and curriculum in comparison to a variety of baselines. 

|Model | Description | #params | Download |
|:------------------------:|:-------------------------------------------:|:------:|:------:|
|sdcup.ch.base | Chinese SDCUP using the  BERT-base architecture | 119M | [sdcup.ch.base](http://alice-open.oss-cn-zhangjiakou.aliyuncs.com/SDCUP/sdcup_base_model.bin-50000) |
|sdcup.ch.large | Chinese SDCUP using the BERT-large architecture | 349M | [sdcup.ch.large](http://alice-open.oss-cn-zhangjiakou.aliyuncs.com/SDCUP/sdcup_large_model.bin-60000) |


## Results
#### sdcup.ch

The results on the Chinese Table Semantic Parsing Dataset

|Model| General | Easy | Medium | Hard |
|:--------------------:|:-------:|:-------:|:-------:|:-------:|
|BERT.Base |85.4 |88.9 |85.7 |81.2 |
|SDCUP.Base |88.7 |90.2|89.2 |85.2 |
|BERT.Large | 87.3 |89.4 |87.9|82.4 |
|SDCUP.Large |90.2 |91.3 |90.8 |86.1 |

#### sdcup.en
The results on the English Table Semantic Parsing Benchmark [SQUALL](https://github.com/tzshi/squall)

|Model | Dev | Test |
|:--------------------:|:-------:|:-------:|
|BERT.Large |64.7 |54.1 |
|SDCUP.Large |71.3 |60.1 |

## Example usage
#### Requirements and Installation
* [PyTorch](https://pytorch.org/) version >= 1.8

* Install other libraries via
```
pip install -r requirements.txt
```

#### Finetune
Please download our pretrained [SDCUP]() model firstly. You can also download other pretrained models from [UER](https://github.com/dbiir/UER-py) or [Hugging Face](https://huggingface.co/uer) model zoo for comparison. We also provide the CBANK dataset which contains 14,625/1,603/1,530 \<Text,SQL\> pairs for training, evaluation and testing.
```
python -u train.py --seed 1 --bS 24 --tepoch 10 --lr 0.001 --lr_bert 0.00001 --table_bert_dir {path_to_downloaded_pretrained_model}  --config_path ./models/bert_base_config.json --vocab_path ./models/google_zh_vocab.txt --data_dir ./data/cbank
```

## Acknowledgement
The finetuning code is implemented based on the [UER](https://github.com/dbiir/UER-py) framework and [sqlova](https://github.com/naver/sqlova). If you use our work, please cite:
```
@article{hui2021improving,
  title={Improving Text-to-SQL with Schema Dependency Learning},
  author={Hui, Binyuan and Shi, Xiang and Geng, Ruiying and Li, Binhua and Li, Yongbin and Sun, Jian and Zhu, Xiaodan},
  journal={arXiv preprint arXiv:2103.04399},
  year={2021}
}
```





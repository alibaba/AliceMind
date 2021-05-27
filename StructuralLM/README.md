# StructuralLM: Structural Pre-training for Form Understanding

[https://arxiv.org/abs/2105.11210](https://arxiv.org/abs/2105.11210)

## Introduction
This work presents a Structural LM model for document image understanding. We introduce a new pre-training approach to jointly leverage cell and layout information from scanned documents. 
## Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|structurallm.en.large | StructuralLM using the BERT-large architecture | 340M | [Structural lm model](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructuralLM/model/structural_lm_models.tar.gz) |

## Example usage
#### Requirements and Installation
* [Tensorflow](https://tensorflow.org/) version == 1.14.0
* Install other libraries via
```
pip install -r requirements.txt
```
Some codes are borrowed from [Bert](https://github.com/google-research/bert)
#### Finetune FUNSD dataset
Download the processed data ([funsd data](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructuralLM/data/funsd_dataset_structurallm.tar.gz))
```
sh finetune_funsd_dataset.sh 
```
#### Other Task
Comming soon

## Citation
If you use our work, please cite:
```
@misc{li2021structurallm,
      title={StructuralLM: Structural Pre-training for Form Understanding}, 
      author={Chenliang Li and Bin Bi and Ming Yan and Wei Wang and Songfang Huang and Fei Huang and Luo Si},
      year={2021},
      eprint={2105.11210},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



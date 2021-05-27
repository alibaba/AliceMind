<p align="center">
  <img src="fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
</p>

--------------------------------------------------------------------------------
This code base is built on top of Fairseq(-py).
Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.


# Pre-trained models

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder and decoder layers trained on 50 languages' monolingual and bilingual corpus | 660M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/model/veco-large.tar.gz)


Download VECO model and save to your path `VECO_MODEL_PATH`


# Getting Started

The following instructions can be used to fine-tune a VECO model on the WMT14 En-Fr and WMT14 En-De datasets.

## Step 1: Preprocess data

You can directly download the preprocessed data via the following urls:
- WMT14 En-Fr [Download](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/data/wmt14.en_de.tar.gz)
- WMT14 En-De [Download](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/data/wmt14.en_fr.tar.gz)

We also provide the bash scripts to download and preprocess the data:
 
```bash
cd examples/veco
bash prepare-wmt14en2de.sh
bash prepare-wmt14en2fr.sh
```
The default saved path of preprocessed data are `examples/veco/data/wmt14.en_de/bin` and `examples/veco/data/wmt14.en_fr/bin`, respectively.


## Step 2: Fine-tune
Next we'll fine-tune a VECO translation model over wmt datasets:

### WMT14 En-Fr
```bash
GPUS=0,1,2,3,4,5,6,7
DATA_DIR=examples/veco/data/wmt14.en_fr/bin  # The path of preprocessed data. 
VECO_MODEL_PATH=saved_model/veco-large/model.pt  # The path of downloaded VECO model.
OUTPUT_DIR=outputs/wmt14.en_fr/finetune  # The directory of saved models and logs during fine-tune.
bash finetune-wmt14en2fr.sh ${GPUS} ${DATA_DIR} ${VECO_MODEL_PATH} ${OUTPUT_DIR}
```

### WMT14 En-De
```bash
GPUS=0,1,2,3,4,5,6,7
DATA_DIR=examples/veco/data/wmt14.en_de/bin  # The path of preprocessed data. 
VECO_MODEL_PATH=saved_model/veco-large/model.pt  # The path of downloaded VECO model.
OUTPUT_DIR=outputs/wmt14.en_de/finetune  # The directory of saved models and logs during fine-tune.
bash finetune-wmt14en2de.sh ${GPUS} ${DATA_DIR} ${VECO_MODEL_PATH} ${OUTPUT_DIR}
```

## Step 3: Inference and Evaluate
Finally we can evaluate our trained model:

### WMT14 En-Fr
```bash
DATA_DIR=examples/veco/data/wmt14.en_fr/bin
MODEL_DIR=outputs/wmt14.en_fr/finetune  # MODEL_DIR is the same as OUTPUT_DIR in the second step.
bash inference-wmt14en2fr.sh ${DATA_DIR} ${MODEL_DIR} 
```

### WMT14 En-De
```bash
DATA_DIR=examples/veco/data/wmt14.en_de/bin
MODEL_DIR=outputs/wmt14.en_de/finetune   # MODEL_DIR is the same as OUTPUT_DIR in the second step.
bash inference-wmt14en2de.sh ${DATA_DIR} ${MODEL_DIR} 
```

## Results on machine translation
The translation results of WMT14 En-Fr and WMT14 En-De:

Model |  WMT14 En-Fr<br>tok/detok-BLEU | WMT14 En-De<br>tok/detok-BLEU
---|---|---
`XLM-R (24 encoder + 6 decoder)` | 43.7/41.1 | 30.8/29.9
`mBART (12 encoder + 12 decoder)` | 43.2/41.0 | 30.0/29.1
`VECO (24 encoder + 6 decoder)` | 44.4/42.0 | 31.5/30.5 

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

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
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

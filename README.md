# AliceMind
#### AliceMind: ALIbaba's Collection of Encoder-decoders from MinD (Machine IntelligeNce of Damo) Lab
This repository provides pre-trained encoder-decoder models and its related optimization techniques developed by Alibaba's MinD (Machine IntelligeNce of Damo) Lab.

The family of AliceMind:
* Pre-trained Models:
     * Language understanding model: [StructBERT](https://github.com/alibaba/AliceMind/tree/main/StructBERT) (```ICLR 2020```)
     * Generative language model: [PALM](https://github.com/alibaba/AliceMind/tree/main/PALM) (```EMNLP 2020```)
     * Cross-lingual language model: [VECO](https://github.com/alibaba/AliceMind/tree/main/VECO) (```ACL 2021```)
     * Cross-modal language model: [StructVBERT](https://github.com/alibaba/AliceMind/tree/main/StructVBERT) (```CVPR 2020 VQA Challenge Runner-up```)
     * Structural language model: [StructuralLM](https://github.com/alibaba/AliceMind/tree/main/StructuralLM) (```ACL 2021```)
     * Chinese language understanding model with multi-granularity inputs: [LatticeBERT](https://github.com/alibaba/AliceMind/tree/main/LatticeBERT) (```NAACL 2021```)
     * Pre-training table model: [SDCUP](https://github.com/alibaba/AliceMind/tree/main/SDCUP) (```Under Review```)
* Fine-tuning Methods:
     * Effective and generalizable fine-tuning method [ChildTuning](https://github.com/alibaba/AliceMind/tree/main/ChildTuning) (```EMNLP 2021```)
* Model Compression:
     * Language model compression methods [ContrastivePruning](https://github.com/alibaba/AliceMind/tree/main/ContrastivePruning) (```AAAI 2022```)

## News
- March, 2021: AliceMind released!
- May, 2021: [VECO](https://arxiv.org/abs/2010.16046) and [StructuralLM](https://arxiv.org/abs/2105.11210) were accepted by ACL 2021.
- September, 2021: The first Chinese pre-training table model [SDCUP](https://arxiv.org/abs/2103.04399) released!
- October, 2021: [ChildTuning](https://arxiv.org/abs/2109.05687) were accepted by EMNLP 2021.
- December, 2021: [ContrastivePruning](https://github.com/alibaba/AliceMind/tree/main/ContrastivePruning) were accepted by AAAI 2022.

## Pre-trained Models
- [**StructBERT**](StructBERT) (March 15, 2021): pre-trained models for **natural language understanding (NLU)**. We extend BERT to a new model, StructBERT, by incorporating language structures into pre-training. Specifically, we pre-train StructBERT with two auxiliary tasks to make the most of the sequential order of words and sentences, which leverage language structures at the word and sentence levels, respectively. "[StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577)" (```ICLR 2020```)

- [**PALM**](PALM) (March 15, 2021): pre-trained models for **natural language generation (NLG)**. We propose a novel scheme that jointly pre-trains an autoencoding and autoregressive language model on a large unlabeled corpus, specifically designed for generating new text conditioned on context. It achieves new SOTA results in several downstream tasks. "[PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation](https://arxiv.org/abs/2004.07159)" (```EMNLP 2020```)

- [**VECO v0**](VECO) (March 15, 2021): pre-trained models for **cross-lingual (x) natural language understanding (x-NLU) and generation (x-NLG)**. VECO (v0) achieves the **new SOTA results** on various cross-lingual understanding tasks of the XTREME benchmark, covering text classification, sequence labeling, question answering, and sentence retrieval.  For cross-lingual generation tasks, it also outperforms all existing cross-lingual models and state-of-the-art Transformer variants on WMT14 English-to-German and English-to-French translation datasets, with gains of up to 1~2 BLEU. “[VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation](https://arxiv.org/abs/2010.16046)" (```ACL 2021```)

- [**StructVBERT**](StructVBERT) (March 15, 2021): pre-trained models for **vision-language understanding**. We propose a new single-stream visual-linguistic pre-training scheme by leveraging multi-stage progressive pre-training and multi-task learning. StructVBERT obtained the 2020 VQA Challenge Runner-up award, and SOTA result on VQA 2020 public Test-standard benchmark (June 2020). "[Talk Slides](StructVBERT/StructVBERT-talk.pdf)" (```CVPR 2020 VQA Challenge Runner-up```).

- [**StructuralLM**](StructuralLM) (March 15, 2021): pre-trained models for **document-image understanding**. We propose a new pre-training approach, StructuralLM, to jointly leverage cell and layout information from scanned documents. The pre-trained StructuralLM achieves new state-of-the-art results in different types of downstream tasks. "[StructuralLM: Structural Pre-training for Form Understanding](https://arxiv.org/abs/2105.11210)" (```ACL 2021```)
- [**LatticeBERT**](LatticeBERT) (March 15, 2021): we propose a novel pre-training paradigm for Chinese — Lattice-BERT which explicitly incorporates word representations with those of characters, thus can model a sentence in a multi-granularity manner. "[Lattice-BERT: Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models](https://arxiv.org/abs/2104.07204)" (`NAACL 2021`)

- [**SDCUP**](SDCUP) (September 6, 2021): pre-trained models for **table understanding**. We design a schema dependency pre-training objective to impose the desired inductive bias into the learned representations for table pre-training. We further propose a schema-aware curriculum learning approach to alleviate the impact of noise and learn effectively from the pre-training data in an easy-to-hard manner. The experiment results on SQUALL and Spider demonstrate the effectiveness of our pre-training objective and curriculum in comparison to a variety of baselines. "[SDCUP: Schema Dependency Enhanced Curriculum Pre-Training for Table Semantic Parsing]()" (```Under Review```) 

## Fine-tuning Methods
- [**ChildTuning**](Child-Tuning) (October 25, 2021): To mitigate the overfitting problem and improve generalization for fine-tuning large-scale PLMs, we
propose a **straightforward yet effective fine-tuning technique**, ChildTuning, which only updates the child network during fine-tuning via strategically masking out the gradients of the non-child network. “[Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning](https://arxiv.org/abs/2109.05687)" (```EMNLP 2021```)

## Model Compression
- [**ContrastivePruning**](ContrastivePruning) (December 17, 2021): 
Contrastive-Pruning is a **general pruning framework under the pre-training and fine-tuning paradigm**, which aims at maintaining both task-specific and task-agnostic knowledge during pruning. CAP is designed as a general framework, compatible with both structured and unstructured pruning. Unified in contrastive learning, CAP encourage the pruned model to learn from the pre-trained model, the snapshots (intermediate models during pruning), and the fine-tuned model, respectively. “[From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression](#)" (```AAAI 2022```)


## Contact Information
**AliceMind Official Website**: [https://nlp.aliyun.com/portal#/alice](https://nlp.aliyun.com/portal#/alice) 

**AliceMind Open Platform**: [https://alicemind.aliyuncs.com](https://alicemind.aliyuncs.com/#/home)

Please submit a GitHub issue if you have want help or have issues using ALICE.

For more information, you can join the ``AliceMind Users Group`` on DingTalk to contact us. The number of the DingTalk group is 35738533.

For other business communications, please contact nlp-support@list.alibaba-inc.com


## *License*

AliceMind is released under the [Apache 2.0 license](LICENSE).

```
Copyright 1999-2020 Alibaba Group Holding Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the following link.

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


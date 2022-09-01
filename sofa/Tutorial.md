
# Tutorial

## Training with huggingface

This is a fully example shown in the [README.md](./README.md)
```python
import sofa
sofa.environ("huggingface")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sofa import apply_child_tuning_to_trainer

# loading datesets
dataset = load_dataset("imdb")

# loading tokenzier
tokenizer = AutoTokenizer.from_pretrained("/tmp/english_sbert-large-std-512", model_max_length=128)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print(tokenizer.__class__)

# tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)
 
# pruned from the original datasets, in order to run test case faster
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# loading the models
model = AutoModelForSequenceClassification.from_pretrained("/tmp/english_sbert-large-std-512", num_labels=2)
print(model)

# training the model
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

##  Pipelines with Huggingface

We also provide a tool to let users write their own pipeline code and use it with transformers pipeline(). In the following case, user can replace `your_finetuned_model_dir` with a location of fine-tuned StructBert model or any huggingface models.

```python
import sofa
sofa.environ("huggingface")
from transformers import pipeline
from sofa.utils import InferenceBase, inject_pipeline
from transformers import AutoModelForSequenceClassification

class MyPipeline(InferenceBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def postprocess(self, outputs, **kwargs):
        return outputs

    def predict(self, inputs, **kwargs):
        return self.model(**inputs)

    def preprocess(self, inputs, **kwargs):
        return self.tokenizer(inputs,
                              add_special_tokens=True,
                              return_tensors="pt",
                              padding=True,
                              truncation="do_not_truncate")


inject_pipeline("my-pipeline-name", MyPipeline, AutoModelForSequenceClassification)

pipe = pipeline("my-pipeline-name", "your_finetuned_model_dir")
res = pipe("some input sentense here")
print(res)

```

Pipelines will be registered into transformers and can be used as other official pipelines.



### Models

#### StructBert

StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding

[https://arxiv.org/abs/1908.04577](https://arxiv.org/abs/1908.04577)

##### Introduction

Please check the model's [readme file](../StructBERT/README.md).

##### Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|structbert.en.large | StructBERT using the BERT-large architecture | 340M | [structbert.en.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/english_sbert-large-std-512.tar.gz) |
|structbert.ch.large | Chinese StructBERT; BERT-large architecture | 330M | [structbert.ch.large](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/chinese_sbert-large-std-512.tar.gz) |

##### How to use

```python
from sofa import SbertModel
SbertModel.from_pretrained(...)
```
###### examples
```shell
# reproduce all of the glue results
sh examples/finetune_sbert_glue.sh
```

##### Results
The results of GLUE & CLUE tasks can be reproduced using the hyperparameters listed in the following "Example usage" section.
###### structbert.en.large
[GLUE benchmark](https://gluebenchmark.com/leaderboard)

|Model| MNLI | QNLIv2 | QQP | SST-2 |
|--------------------|-------|-------|-------|-------|
|structbert.en.large |86.62% |93.34% |91.99% |93.00% |

###### structbert.ch.large
[CLUE benchmark](https://www.cluebenchmarks.com/)

|Model | TNEWS | CSL | CMNLI | CLUEWSC | IFLYTEK | OCNLI | AFQMC | CLUENER(F1) |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|structbert.ch.large |61.34% |83.43% |84.09% |90.79% |62.79% | 80.70% | 77.32% | 77.98% |

---

#### Palm

PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation

[Paper Link](https://arxiv.org/abs/2004.07159)

##### Introduction

Please check the model's [readme file](../PALM/README.md).

##### Pre-trained models

|Model | Description | #params | Download |
---|---|---|---
palm.ch.base | PALM using a 12-layer encoder and a 12-layer decoder on Chinese training data | 217M | [chinese-palm-base.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/chinese-palm-base.tar.gz)

##### How to use

To use the palm model in the code, you need to import sofa toolkit first
```python
import sofa
sofa.environ("huggingface")
```
If you only need to use the palm's backbone, you can use the following way to call

output by default:

prediction_scores, pooled_output

With parameter is_infer=True, the output will be:

prediction_scores, pooled_output，sequence_output

```python
from sofa import PalmModel
palm_model = PalmModel.from_pretrained(pretrain_model_path)
# Another way, building model from AutoModel of transformers
# model_type: "palm" in config.json or "palm" in str(pretrain_model_path)
from transformers import AutoModel
palm_model = AutoModel.from_pretrained(pretrain_model_path)

palm_model(**batched_data, is_infer=False)
```
It is recommended to use PalmForConditionalGeneration in NLG tasks
```python
from sofa import PalmForConditionalGeneration
model = PalmForConditionalGeneration.from_pretrained(pretrain_model_path)
# or
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path)
```
The form of input data: "sentence1"\t"sentence2" (without quotation marks) is one line, a .txt file consisting of one or more lines.

generation.py also provides a method to use Trainer Callback to output the generation results at the end of each epoch and calculate the bleu-1/2/3/4 and rouge indicators, see the run_generation_hf function for details.
```python
# Use TextGenerator (in generation.py)
beam_generator = TextGenerator(model, tokenizer, tokenizer.vocab)
# Or finetune with one line of code
from sofa import run_generation_hf
run_generation_hf(pretrain_model_path,
                  task_type="generation",
                  train_file_path=train_file_path,
                  dev_file_path=dev_file_path)
```
###### examples
An example that can be run immediately without worrying about data and model loading
```shell
# palm-NLG finetune
sh examples/finetune_palm.sh
```

##### Results

###### palm.ch.base
[DuReader benchmark](https://github.com/baidu/DuReader)

[CLGE benchmark](https://github.com/CLUEbenchmark/CLGE)

|palm.ch.base | DuReaderQG | DuReader-Robust | LCSTS |
|--------------------|-------|-------|-------|
|BLEU-1 | 0.5863 | 0.5560 | 0.3296 |
|BLEU-4 | 0.3982 | 0.3553 | 0.1716 |
|ROUGE-L | 0.6027 | 0.5661 | 0.3293 |

---

#### Veco

VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation

[Paper Link](https://arxiv.org/abs/2010.16046)

##### News
- May, 2021: [VECO](https://arxiv.org/abs/2010.16046) was accepted by ACL 2021.
- Mar, 2021: VECO ranks first at the [XTREME](https://sites.research.google/xtreme/) leaderboard.

##### Introduction

Please check the model's [readme file](../VECO/README.md).

##### Pre-trained models

###### Pre-trained models for Multilingual NLU tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder layers trained on 50 languages' monolingual and bilingual corpus | 550M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLU/model/xtreme-released-veco.tar.gz)


###### Pre-trained models for Multilingual NLG tasks

Model | Description | # Params | Download
---|---|---|---
`veco_large` | VECO model with 24 encoder and decoder layers trained on 50 languages' monolingual and bilingual corpus | 660M | [veco-large.tar.gz](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLG/model/veco-large.tar.gz)

##### How to use

```python
from sofa import VecoModel
VecoModel.from_pretrained(...)
```
###### examples
```shell
# NLU finetune
sh examples/finetune_veco.sh
# veco for NLG is comming soon! :)
```

##### Results

###### Results of Multilingual NLU tasks

[XTREME](https://sites.research.google/xtreme/) is one of the most representative massively multilingual benchmark.

The results of XNLI, XQuAD and Tatoeba task can be reproduced using the default hyperparameters listed in `.sh` file.

|Model| XNLI<br>(Acc) | XQuAD<br>(F1/EM) | Tatoeba<br>(Acc) |
|--------------------|-------|-------|-------|
|veco.large | 79.9 | 77.5/61.9 | 75.1 |

###### Results of Multilingual NLG tasks

Results on custom machine translation datasets.

Model |  WMT14 En-Fr<br>tok/detok-BLEU | WMT14 En-De<br>tok/detok-BLEU
---|---|---
`XLM-R (24 encoder + 6 decoder)` | 43.7/41.1 | 30.8/29.9
`mBART (12 encoder + 12 decoder)` | 43.2/41.0 | 30.0/29.1
`VECO (24 encoder + 6 decoder)` | 44.4/42.0 | 31.5/30.5

### Fine-tuning Technologys

#### Child-Tuning

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

You can use Child-Tuning like:

Use an optimizer dependently:
```python
from sofa.utils import ChildTuningAdamW
optimizer = ChildTuningAdamW(...)
# This is a custom optimizer based on AdamW, almost all the params are the same except:
# reserve_p: a reserve probability of gradiants, defaults to 0.2.
# mode: child-tuning mode, defaults to "ChildTuning-F".
```
Commonly, tasked related child-tuning "ChildTuning-D" is better than "ChildTuning-F". 
If you want to use "ChildTuning-D", you need to pass "ChildTuning-D" to the "mode" parameter.
Note: "ChildTuning-D" is task related, so you cannot use it in apply_child_tuning function.
Usually, child-tuning shows effects when training features are less than 100,000, 
especially when less than 10,000.

Use in huggingface:
```python
import sofa
sofa.environ("huggingface")
from sofa.utils import apply_child_tuning, apply_child_tuning_to_trainer
from transformers import Trainer
# apply child tuning to single trainer instance:
trainer = Trainer(...)
apply_child_tuning_to_trainer(trainer)
# or apply Child-Tuning to the Trainer class, so you can use it in all Trainer instances.
apply_child_tuning()
```


# Sparse BERT/RoBERTa using PST

This folder contains the implementation of PST in BERT/RoBERTa using the Python package pst and steps to replicate the results in our paper.
Our implementation is based on the fine-tuning code for BERT in Hugging Face.

## Getting Started

### Replicating Our Result on BERT

```bash
TASK_NAME=sst2

python3 run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name ${TASK_NAME} \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 20 \
  --seed 43 \
  --output_dir tmp/bert
```


### Replicating Our Result on RoBERTa

```bash
TASK_NAME=sst2

python3 run_glue.py \
  --model_name_or_path roberta-base \
  --task_name ${TASK_NAME} \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --seed 43 \
  --output_dir tmp/roberta
```

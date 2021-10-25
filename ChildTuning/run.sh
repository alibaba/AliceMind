#!/bin/bash

GPU=0
OUTPUT_DIR=output
REVERSE_P=0.3
MODE=ChildTuning-D # choose from ['ChildTuning-F', 'ChildTuning-D']
TASK=rte
MODEL=bert-large-cased
BSZ=16
EPOCH=3.0
LR=4e-5
SEED=42

CUDA_VISIBLE_DEVICES=${GPU} python run_glue.py \
--model_name_or_path ${MODEL} \
--task_name ${TASK} \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size ${BSZ} \
--learning_rate ${LR} \
--num_train_epochs ${EPOCH} \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--output_dir ${OUTPUT_DIR} \
--seed ${SEED} \
--save_total_limit 1 \
--save_steps 30000 \
--reserve_p ${REVERSE_P} \
--mode ${MODE}


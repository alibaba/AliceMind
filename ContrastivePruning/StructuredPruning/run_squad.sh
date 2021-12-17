#!/bin/bash

# basic
GPU=0
OUTPUT_DIR=cap
TASK_NAME=squad
MODEL=bert-base-uncased
MAX_SEQ_LENGTH=384
LR=3e-5
BSZ=16
WEIGHT_DECAY=0.01
PRUNE_SEQUENCE=10,20,30,40,50,60,70,80,90

# contrastive
CONTRASTIVE_TEMPERATURE=0.3
EXTRA_EXAMPLES=4096
ALIGNREP=cls
RETRAIN_EPOCH=2.0
CE_LOSS_WEIGHT=0.1
CL_UNSUPERVISED_LOSS_WEIGHT=0.3

# distill
TEACHER_PATH=../teacher/squad
DISTILL_LOSS_WEIGHT=0.9
DISTILL_TEMPERATURE=1.0

CUDA_VISIBLE_DEVICES=${GPU} python run_squad.py \
--model_name_or_path ${MODEL} \
--dataset_name squad \
--do_train \
--do_eval \
--max_seq_length ${MAX_SEQ_LENGTH} \
--doc_stride 128 \
--per_device_train_batch_size ${BSZ} \
--learning_rate ${LR} \
--warmup_ratio 0.1 \
--weight_decay ${WEIGHT_DECAY} \
--output_dir ${OUTPUT_DIR}/${TASK_NAME} \
--save_total_limit 1 \
--save_steps 30000 \
--do_prune \
--prune_percent ${PRUNE_SEQUENCE} \
--normalize_pruning_by_layer \
--subset_ratio 1.0 \
--retrain_num_train_epochs ${RETRAIN_EPOCH} \
--use_contrastive_loss \
--contrastive_temperature ${CONTRASTIVE_TEMPERATURE} \
--ce_loss_weight ${CE_LOSS_WEIGHT} \
--cl_unsupervised_loss_weight ${CL_UNSUPERVISED_LOSS_WEIGHT} \
--extra_examples ${EXTRA_EXAMPLES} \
--alignrep ${ALIGNREP} \
--use_distill \
--teacher_path ${TEACHER_PATH} \
--distill_temperature ${DISTILL_TEMPERATURE} \
--distill_loss_weight ${DISTILL_LOSS_WEIGHT} 

rm -r ${OUTPUT_DIR}/${TASK_NAME}/checkpoint*
rm -r ${OUTPUT_DIR}/${TASK_NAME}/*.bin

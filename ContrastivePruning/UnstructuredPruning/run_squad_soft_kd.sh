#!/bin/bash

# basic
GPU=0
OUTPUT=soft-cap
TASK=squad
DATA_DIR=../data/squad
MODEL=bert-base-uncased
BATCH=16
EPOCH=10
LR=3e-5

# pruning
METHOD=sigmoied_threshold
MASK_LR=1e-2
WARMUP=5400
INITIAL_TH=0
FINAL_TH=0.1
REGULARIZATION=l1
FINAL_LAMBDA=2000

# contrastive
CONTRASTIVE_TEMPERATURE=0.1
EXTRA_EXAMPLES=4096
ALIGNREP=cls # ['cls', 'mean-pooling']
CL_UNSUPERVISED_LOSS_WEIGHT=0.5

# distill
TEACHER_TYPE=bert
TEACHER_PATH=../teacher/squad
CE_LOSS_WEIGHT=0.1
DISTILL_LOSS_WEIGHT=0.9


CUDA_VISIBLE_DEVICES=${GPU} python masked_run_squad.py \
    --output_dir ${OUTPUT}/${FINAL_LAMBDA}/${TASK} \
    --data_dir ${DATA_DIR} \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path ${MODEL} \
    --per_gpu_train_batch_size ${BATCH} \
    --warmup_steps ${WARMUP} \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} --mask_scores_learning_rate ${MASK_LR} \
    --initial_threshold ${INITIAL_TH} --final_threshold ${FINAL_TH} \
    --initial_warmup 1 --final_warmup 2 \
    --pruning_method ${METHOD} --mask_init constant --mask_scale 0.0 \
    --regularization ${REGULARIZATION} \
    --final_lambda ${FINAL_LAMBDA} \
    --save_steps 30000 \
    --use_contrastive_loss \
    --contrastive_temperature ${CONTRASTIVE_TEMPERATURE} \
    --cl_unsupervised_loss_weight ${CL_UNSUPERVISED_LOSS_WEIGHT} \
    --extra_examples ${EXTRA_EXAMPLES} \
    --alignrep ${ALIGNREP} \
    --use_distill \
    --teacher_name_or_path ${TEACHER_PATH} \
    --teacher_type ${TEACHER_TYPE} \
    --ce_loss_weight ${CE_LOSS_WEIGHT} \
    --distill_loss_weight ${DISTILL_LOSS_WEIGHT} 
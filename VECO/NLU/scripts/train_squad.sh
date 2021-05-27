#!/bin/bash
# Copyright 2020 Google, DeepMind and Alibaba inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to train a model on SQuAD v1.1 train data.

REPO=$PWD
REPO=$PWD
MODEL_PATH=${1:-"$PWD/VECO"}
GPU=${2:-0}
LR=${3:-"2e-5"}
EPOCH=${4:-2}
TOTAL_BATCH_SIZE=${5:-16}
OUT_DIR=${6:-"$REPO/outputs"}
DATA_DIR=${7:-"$REPO/download/"}

TRAIN_FILE=${DATA_DIR}/squad/train-v1.1.json
DEV_FILE=${DATA_DIR}/squad/dev-v1.1.json
TEST_DIR=${DATA_DIR}/xquad/
MAXL=384
TASK='squad'

dir_array=(${MODEL_PATH//// })
OUTPUT_DIR="$OUT_DIR/$TASK/${dir_array[-1]}.LR${LR}-Epoch${EPOCH}-BatchSize${TOTAL_BATCH_SIZE}"
LOG_FILE=$OUTPUT_DIR/train.log

echo "MODEL_PATH: $MODEL_PATH"
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "log to file: $LOG_FILE"

BATCH_SIZE=2
GRAD_ACC=`expr $TOTAL_BATCH_SIZE / $BATCH_SIZE`
LANGS="en,es,de,el,ru,tr,ar,vi,th,zh,hi"

# train
CUDA_VISIBLE_DEVICES=$GPU python $PWD/third_party/run_squad.py \
  --model_name_or_path ${MODEL_PATH} \
  --do_train \
  --do_predict \
  --save_only_best_checkpoint \
  --predict_test_dataset \
  --train_file ${TRAIN_FILE} \
  --dev_file ${DEV_FILE} \
  --predict_file ${TEST_DIR} \
  --per_gpu_eval_batch_size 16 \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCH} \
  --max_seq_length $MAXL \
  --doc_stride 64 \
  --save_steps 200 \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --warmup_steps 500 \
  --output_dir ${OUTPUT_DIR} \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang en \
  --eval_lang en \
  --log_file $LOG_FILE \
  --predict_languages ${LANGS} \
  --overwrite_cache


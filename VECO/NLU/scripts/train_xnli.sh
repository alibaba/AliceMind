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

REPO=$PWD
MODEL_PATH=${1:-"$PWD/VECO"}
GPU=${2:-0}
LR=${3:-"2e-5"}
EPOCH=${4:-2}
TOTAL_BATCH_SIZE=${5:-64}
OUT_DIR=${6:-"$REPO/outputs"}
DATA_DIR=${7:-"$REPO/download/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
MAXL=128
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"

BATCH_SIZE=2
GRAD_ACC=`expr $TOTAL_BATCH_SIZE / $BATCH_SIZE`

dir_array=(${MODEL_PATH//// })
OUTPUT_DIR="$OUT_DIR/$TASK/${dir_array[-1]}.LR${LR}-Epoch${EPOCH}-BatchSize${TOTAL_BATCH_SIZE}"
LOG_FILE=$OUTPUT_DIR/train.log

echo "MODEL_PATH: $MODEL_PATH"
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "log to file: $LOG_FILE"

python $PWD/third_party/run_classify.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK \
  --do_train \
  --do_predict \
  --data_dir $DATA_DIR/$TASK/ \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 128 \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $OUTPUT_DIR/ \
  --save_steps 500 \
  --eval_all_checkpoints \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --overwrite_cache \
  --log_file $LOG_FILE \
  --predict_languages $LANGS

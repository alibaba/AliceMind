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

BATCH_SIZE=16
TASK_TYPE=generation
CUDA_VISIBLE=0

TIME=$(date "+%Y-%m%d-%H%M%S")
REPO="$PWD/palm"
MODEL_PATH="$REPO/model"
OUT_DIR="$REPO/outputs/${TASK_TYPE:0:7}_$TIME"
DATA_DIR="$REPO/data"
DATA_TYPE="weather" # "weather" or "dureaderqg" or "dureader_robust" or "lcsts" 

mkdir -p $MODEL_PATH
mkdir -p $OUT_DIR
mkdir -p $DATA_DIR

# download model
if [ ! -d $MODEL_PATH/chinese_palm_base ]; then
  echo "Downloading pretrained model to $MODEL_PATH"
  wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/chinese-palm-base.tar.gz -P $MODEL_PATH
  mkdir -p $MODEL_PATH/chinese_palm_base
  tar xvf $MODEL_PATH/chinese-palm-base.tar.gz -C $MODEL_PATH/chinese_palm_base
  rm $MODEL_PATH/chinese-palm-base.tar.gz
fi

# download dataset
# if [ ! -f $DATA_DIR/weather_train.txt ]; then
#   echo "Downloading dataset to $DATA_DIR"
#   wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/weather_train.txt -P $DATA_DIR
# fi
# if [ ! -f $DATA_DIR/weather_dev.txt ]; then
#   wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/weather_dev.txt -P $DATA_DIR
if [ ! -f $DATA_DIR/dev.txt ]; then
  python utils/generation_data.py $DATA_DIR $DATA_TYPE
fi

echo "OUTPUT_DIR: $OUT_DIR"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE python finetune_with_huggingface.py \
        --pretrain_model_path $MODEL_PATH/chinese_palm_base \
        --train_file_path $DATA_DIR/train.txt \
        --output_dir $OUT_DIR \
        --task_type $TASK_TYPE \
        --dev_file_path $DATA_DIR/dev.txt \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --num_train_epochs 10 \
        --max_sequence_length 128 \
        --save_strategy steps \
        --save_total_limit 1 \

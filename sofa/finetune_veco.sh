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

BATCH_SIZE=2

TIME=$(date "+%Y-%m%d-%H%M%S")

REPO="$PWD/veco"
MODEL_PATH="$REPO/model"
OUT_DIR="$REPO/outputs/single_$TIME"

mkdir -p $MODEL_PATH
mkdir -p $OUT_DIR

# download model
if [ ! -d $MODEL_PATH/xtreme-released-veco ]; then
  echo "Downloading pretrained model at $MODEL_PATH"
  wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/VECO/NLU/model/xtreme-released-veco.tar.gz -P $MODEL_PATH
  tar xvf $MODEL_PATH/xtreme-released-veco.tar.gz -C $MODEL_PATH
  rm $MODEL_PATH/xtreme-released-veco.tar.gz
fi

# download dataset

echo "OUTPUT_DIR: $OUT_DIR"

python3 finetune_with_huggingface.py \
        --pretrain_model_path $MODEL_PATH/xtreme-released-veco \
        --output_dir $OUT_DIR \
        --task_type single_label_classification \
        --dataset_name xnli \
        --train_dataset_config en \
        --eval_dataset_config ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
        --learning_rate 2e-5 \
        --seed 42 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 128 \
        --num_train_epochs 2 \
        --eval_steps 500 \
        --max_sequence_length 128 \
        --pair 1 \
        --load_best_model_at_end=True \
        --save_strategy steps \
        --save_total_limit 1 \
        --evaluation_strategy steps \
        --gradient_accumulation_steps 32


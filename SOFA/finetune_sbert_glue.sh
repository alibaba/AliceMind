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

BATCH_SIZE=32
GLUE_DATASET=sst2  # mnli/qqp/qnli
TASK_TYPE=single_label_classification

TIME=$(date "+%Y-%m%d-%H%M%S")
REPO="$PWD/sbert"
MODEL_PATH="$REPO/model"
OUT_DIR="$REPO/outputs/${TASK_TYPE:0:7}_$TIME"

mkdir -p $MODEL_PATH $OUT_DIR

# download model
if [ ! -d $MODEL_PATH/english_sbert-large-std-512 ]; then
  echo "Downloading pretrained model to $MODEL_PATH"
  wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/english_sbert-large-std-512.tar.gz -P $MODEL_PATH
  mkdir -p $MODEL_PATH/english_sbert-large-std-512
  tar xvf $MODEL_PATH/english_sbert-large-std-512.tar.gz -C $MODEL_PATH/english_sbert-large-std-512
  rm $MODEL_PATH/english_sbert-large-std-512.tar.gz
fi

echo "OUTPUT_DIR: $OUT_DIR"

# dataset args
declare -A pair=([mnli]=1 [qnli]=1 [qqp]=1 [sst2]=0)

python finetune_with_huggingface.py \
        --pretrain_model_path $MODEL_PATH/english_sbert-large-std-512 \
        --dataset_name glue \
        --train_dataset_config $GLUE_DATASET \
        --eval_dataset_config $GLUE_DATASET \
        --output_dir $OUT_DIR \
        --task_type $TASK_TYPE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 128 \
        --num_train_epochs 3 \
        --eval_steps 500 \
        --max_sequence_length 128 \
        --pair ${pair[$GLUE_DATASET]} \
        --load_best_model_at_end=True \
        --save_strategy steps \
        --save_strategy steps \
        --save_total_limit 1 \
        --evaluation_strategy steps \
        --weight_decay 1
# With the correct installation of the apex, the combination below could be used for faster training.
#         --half_precision_backend apex   --fp16_opt_level O1

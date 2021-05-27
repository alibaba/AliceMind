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
LAYER=${3:-18}
OUT_DIR=${4:-"$REPO/outputs/"}
DATA_DIR=${5:-"$REPO/download/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='tatoeba'
TL='en'
MAXL=512
LC=""
DIM=1024
NLAYER=24

dir_array=(${MODEL_PATH//// })
OUTPUT_DIR="$OUT_DIR/$TASK/${dir_array[-1]}.LAYER${LAYER}"
LOG_FILE=$OUTPUT_DIR/run.log
echo "OUTPUT_DIR: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
LANGS="ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu"
python $PWD/third_party/run_retrieval.py \
  --model_name_or_path $MODEL_PATH \
  --embed_size $DIM \
  --batch_size 100 \
  --src_languages $LANGS \
  --tgt_language en \
  --data_dir $DATA_DIR/$TASK/ \
  --max_seq_length $MAXL \
  --output_dir $OUTPUT_DIR \
  --log_file LOG_FILE \
  --num_layers $NLAYER \
  --dist cosine $LC \
  --specific_layer $LAYER


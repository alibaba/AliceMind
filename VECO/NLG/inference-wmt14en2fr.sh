#!/bin/bash

DATA_DIR=${1:-"data/wmt14.en_fr/bin"}
#DATA_DIR=${1:-"data/wmt14.en_fr/bin-resized_dict"}
MODEL_DIR=${2:-"outputs/wmt14.en_fr/finetune"}
OUTPUTS_DIR=${3:-"outputs/wmt14.en_fr/inference"}

# export CUDA_VISIBLE_DEVICES=0

echo "Average last 10 checkpoints..."
echo "Save log to: ${OUTPUTS_DIR}/avergaed_checkpoint.log"
python3 -u scripts/average_checkpoints.py \
  --inputs ${MODEL_DIR} \
  --num-update-checkpoints 10 \
  --checkpoint-upper-bound 100000 \
  --output ${MODEL_DIR}/avergaed_checkpoint.pt \
  | tee ${OUTPUTS_DIR}/avergaed_checkpoint.log;


echo "Start to inference..."
echo "Save log to: ${OUTPUTS_DIR}/inference.log"

if [[ $DATA_DIR == *"resized_dict" ]] ; then
  MODEL_DIR=$MODEL_DIR-resized_dict
  OUTPUTS_DIR=$OUTPUTS_DIR-resized_dict
  RESIZED_DICT_MAP="examples/veco/data/wmt14.en_fr/dict/resized_dict_map.txt"
  EXTRA_ARG="--resized-dict-map ${RESIZED_DICT_MAP} "
else
  EXTRA_ARG=""
fi

mkdir -p $OUTPUTS_DIR

python3 -u generate.py ${DATA_DIR} --path ${MODEL_DIR}/avergaed_checkpoint.pt \
  --task translation_from_pretrained_veco $EXTRA_ARG \
  --source-lang en --target-lang fr --gen-subset test \
  --batch-size 128  --beam 5 \
  --remove-bpe sentencepiece --sacrebleu  \
  | tee ${OUTPUTS_DIR}/inference.log


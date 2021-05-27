#!/bin/bash

GPUS=${1:-"0,1,2,3,4,5,6,7"}
DATA_DIR=${2:-"examples/veco/data/wmt14.en_de/bin"}
#DATA_DIR=${2:-"examples/veco/data/wmt14.en_de/bin-resized_dict"}
INIT_MODEL=${3:-"saved_model/veco-large/model.pt"}
OUTPUT_DIR=${4:-"outputs/wmt14.en_de/finetune"}


if [[ $DATA_DIR == *"resized_dict" ]] ; then
  OUTPUT_DIR=${OUTPUT_DIR}-resized_dict
  RESIZED_DICT_MAP="examples/veco/data/wmt14.en_de/dict/resized_dict_map.txt"
  EXTRA_ARG="--resized-dict-map ${RESIZED_DICT_MAP} "
else
  EXTRA_ARG=""
fi

mkdir -p $OUTPUT_DIR

# Batch size: 64k = 16 gpu-32G * 4k max-tokens * 1 update-freq = 8 gpu-32G * 4k max-tokens * 2 update-freq

echo "Start training on GPUS: ${GPUS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python3 -u train.py $DATA_DIR \
  --fp16 \
  --task=translation_from_pretrained_veco \
  --criterion=label_smoothed_cross_entropy \
  --label-smoothing=0.2 \
  --arch=veco_large \
  --share-all-embeddings \
  --keep-decoder-layers=18,19,20,21,22,23 \
  --source-lang=en \
  --target-lang=de \
  --layernorm-embedding \
  --dataset-impl=mmap \
  --num-workers=16 \
  --restore-file=$INIT_MODEL \
  --max-tokens=4096 \
  --update-freq=2 \
  --lr-scheduler=polynomial_decay \
  --lr=1e-4 \
  --min-lr=-1 \
  --warmup-updates=16000 \
  --total-num-update=100000 \
  --patience=30 \
  --optimizer=adam \
  --adam-eps=1e-08 \
  --adam-betas='(0.9, 0.999)' \
  --clip-norm=0.1 \
  --weight-decay=0.01 \
  --dropout=0.3 \
  --attention-dropout=0.1 \
  --max-target-positions=512 \
  --max-source-positions=512 \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --reset-lr-scheduler \
  --save-interval=5 \
  --save-interval-updates=5000 \
  --keep-interval-updates=10 \
  --no-save-optimizer-state \
  --keep-best-checkpoints=1 \
  --seed=222 \
  --log-format=simple \
  --log-interval=100 \
  --ddp-backend=no_c10d \
  --save-dir=$OUTPUT_DIR $EXTRA_ARG \
  | tee $OUTPUT_DIR/finetune.log


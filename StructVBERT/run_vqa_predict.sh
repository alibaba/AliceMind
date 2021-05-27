#!/usr/bin/env bash

# The name of this experiment.
name=run-vqa-structvbert-base

# Save logs and models under snap/vqa; make backup.
model_ckpt=output/$name

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PYTHONPATH:./ \
    python tasks/vqa.py \
    --bert_model pretrained_model \
    --tiny --train train --valid "" --test test \
    --llayers 0 --xlayers 12 --rlayers 0 \
    --one_stream \
    --use_npz \
    --image_hdf5_file data/mscoco_imgfeat/train_npz,data/mscoco_imgfeat/test_npz \
    --load ${model_ckpt}/BEST \
    --num_workers 2 \
    --amp_type O1 \
    --padding \
    --max_objects 100 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 3 \
    --tqdm --output $output

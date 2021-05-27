#!/usr/bin/env bash

# The name of this experiment.
name=run-vqa-structvbert-base

# Save logs and models under snap/vqa; make backup.
output=output/$name
mkdir -p $output

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PYTHONPATH:./ \
    python tasks/vqa.py \
    --bert_model pretrained_model \
    --train train,nominival --valid minival  \
    --llayers 0 --xlayers 12 --rlayers 0 \
    --one_stream \
    --use_npz \
    --image_hdf5_file data/mscoco_imgfeat/train_npz,data/mscoco_imgfeat/valid_npz \
    --loadLXMERTQA pretrained_model/pytorch_model.bin \
    --num_workers 2 \
    --amp_type O1 \
    --padding \
    --max_objects 100 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 3 \
    --tqdm --output $output

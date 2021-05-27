#!/usr/bin/env bash

# The name of this experiment.
name=run-nlvr2-structvbert-base-test

# Save logs and models under snap/nlvr2; Make backup.
output=output/$name
mkdir -p $output

# See run/Readme.md for option details.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./ \
    python tasks/nlvr2.py \
    --bert_model pretrained_model \
    --train train --valid test \
    --llayers 0 --xlayers 12 --rlayers 0 \
    --one_stream \
    --use_npz \
    --image_hdf5_file data/nlvr2_imgfeat/nlvr2_train_npz,data/nlvr2_imgfeat/nlvr2_test_npz \
    --loadLXMERT pretrained_model/pytorch_model.bin \
    --num_workers 2 \
    --amp_type O1 \
    --padding \
    --max_objects 100 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 3 \
    --tqdm --output $output

#--image_hdf5_file /home/ym119608/vqa-lxmert/data/nlvr2_imgfeat/nlvr2_train_obj100_npz,/home/ym119608/vqa-lxmert/data/nlvr2_imgfeat/nlvr2_valid_obj100_npz,/home/ym119608/vqa-lxmert/data/nlvr2_imgfeat/nlvr2_test_obj100_npz \

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=2224 \
    --use_env retrieval_img_mplug.py \
    --config ./configs/retrieval_coco_mplug_large.yaml \
    --output_dir output/retrieval_coco_mplug_large \
    --checkpoint ./mplug_large.pth \
    --do_two_optim \
    --do_amp
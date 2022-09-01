python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=3224 \
    --use_env videoqa_mplug.py \
    --config ./configs/videoqa_mplug_base_msvd.yaml \
    --checkpoint vqa_mplug_base/vqa_best_mplug_base.pth \
    --output_dir output/videoqa_mplug_base_msvd
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=2224 \
    --use_env retrieval_vid_mplug.py \
    --config ./configs/retrieval_msrvtt_mplug_large.yaml \
    --checkpoint ./mplug_large.pth
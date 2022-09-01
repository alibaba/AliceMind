# uni training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=3222  --use_env grounding_mplug.py \
#     --config configs/grounding_mplug_large.yaml \
#     --dataset vg_uni \
#     --output_dir ./output/vg_large_uni \
#     --checkpoint ./mplug_large.pth \
#     --do_two_optim

# refcoco
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=3222  --use_env grounding_mplug.py \
#     --config configs/grounding_mplug_large.yaml \
#     --dataset vg_unc \
#     --output_dir ./output/vg_large_unc \
#     --checkpoint ./output/vg_large_uni/checkpoint_best.pth  \
#     --do_two_optim
# # refcoco+
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=3222  --use_env grounding_mplug.py \
#     --config configs/grounding_mplug_large.yaml \
#     --dataset vg_unc+ \
#     --output_dir ./output/vg_large_unc+ \
#     --checkpoint ./output/vg_large_uni/checkpoint_best.pth  \
#     --do_two_optim
# # gref_umd
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=3222  --use_env grounding_mplug.py \
#     --config configs/grounding_mplug_large.yaml \
#     --dataset vg_gref_umd \
#     --output_dir ./output/vg_large_gref_umd \
#     --checkpoint ./output/vg_large_uni/checkpoint_best.pth  \
#     --do_two_optim

# eval refcoco
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=3222  --use_env grounding_mplug.py \
    --config configs/grounding_mplug_large.yaml \
    --dataset vg_unc \
    --output_dir ./output/vg_large_unc \
    --eval_checkpoint ./output/vg_large_unc/checkpoint_best.pth \
    --do_two_optim --evaluate

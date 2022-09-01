# deepspeed==0.5.8

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=3224 \
    --use_env vqa_mplug.py \
    --config ./configs/vqa_mplug_large.yaml \
    --output_dir output/vqa_mplug_large_4m \
    --checkpoint ./mplug_large_v2.pth \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json 

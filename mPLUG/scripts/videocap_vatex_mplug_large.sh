pip install colorlog
apt-get update
apt-get install default-jdk
apt-get install default-jre
pip install git+git://github.com/j-min/language-evaluation@master
python -c "import language_evaluation; language_evaluation.download('coco')"


python -m torch.distributed.launch --nproc_per_node=8 --master_port=3223  --use_env videocap_mplugx.py \
    --config ./configs/videocap_vatext_mplug_large.yaml \
    --output_dir output/videocap_vatext_mplug_larg \
    --checkpoint mplug_large.pth \
    --do_two_optim \
    --min_length 15 \
    --beam_size 10 \
    --max_length 25 \
    --max_input_length 25 \
    --evaluate \
    --do_amp

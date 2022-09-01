#!/bin/sh
set -x

project_dir=$PWD
data_dir="$project_dir/data"
log_dir="$project_dir/logs"
model_dir="$project_dir/model"
save_dir="$project_dir/save"

mkdir -p $log_dir $save_dir $model_dir $data_dir

# Data config
task=dureader_robust
task_type=generation

# Download model
if [ ! -d $model_dir/plug_model ]; then
  echo "Downloading pretrained model to $model_dir"
  wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PLUG/plug_model.tar -P $model_dir
  mkdir -p $model_dir/plug_model
  tar xvf $model_dir/plug_model.tar -C $model_dir/plug_model
  rm $model_dir/plug_model.tar
fi

# Download data
if [ ! -f $data_dir/$task/dev.txt ]; then
  mkdir -p $data_dir/$task
  python3 utils/generation_data.py $data_dir/$task $task
fi
  
# Distributed training config
_world_size=1
_model_parallel=8
_ckpt=1
_zero='ds_zero-offload_10B_config.json'

RANDOM=1805

# Training config
_dropout=0.1
_batch_size=32
_lr=1e-5
_epoch=5
_warm_up=0.01

_load_iter=28000
_attn_separate=0

_other_tags=$(date +%y%m%d_%H%M)

max_len=384
tgt_len=30
min_len=5

if [ ${_ckpt} = 1 ]
then
_ckpt="--checkpoint-activations --deepspeed-activation-checkpointing"
else
_ckpt=" "
fi

if [ ${_model_parallel} = 16 ]
then
_hostfile="--hostfile=./hostfile"
else
_hostfile=" "
fi

tags=${task}_h${hidden}_p${_model_parallel}_n${_world_size}_bs${_batch_size}_ep${_epoch}_lr${_lr}_${_other_tags}

# export NCCL_DEBUG=INFO && export NCCL_IB_GID_INDEX=3 && export NCCL_IB_HCA=mlx5 &&
export PYTHONPATH="${PYTHONPATH}:$PWD" && \
deepspeed \
    ${_hostfile} \
    --num_gpus=8 \
    --num_nodes=${_world_size} \
    ${project_dir}/finetune_plug.py \
    --model-parallel-size ${_model_parallel} \
    --min-length ${min_len} \
    --tgt-length ${tgt_len} \
    --deep-init \
    --attention-dropout ${_dropout} \
    --hidden-dropout ${_dropout} \
    --fp32-layernorm \
    --shuffle \
    --seed ${RANDOM} \
    --save ${save_dir}/ckpt_${tags} \
    --save-interval 100 \
    --downstream-dataset \
    --batch-size ${_batch_size} \
    --eval-batch-size ${_batch_size} \
    --seq-length ${max_len} \
    --max-position-embeddings 2048 \
    --log-interval 5 \
    --pre-load \
    --load-iteration ${_load_iter} \
    --load ${model_dir}/plug_model \
    --eval-iters 100 \
    --eval-interval 500 \
    --train-data ${data_dir}/${task} \
    --dev-data ${data_dir}/${task} \
    --task-name ${task} \
    --task_type ${task_type} \
    --tokenizer-type BertWordPieceTokenizer \
    --tokenizer-model-type ${project_dir}/ch_vocab.txt \
    --distributed-backend nccl \
    --lr ${_lr} \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-epochs ${_epoch} \
    --warmup ${_warm_up} \
    --cpu-optimizer \
    ${_ckpt} \
    --fp16 \
    --deepspeed \
    --deepspeed_config ${project_dir}/$_zero >> ${log_dir}/${tags}.log 2>&1


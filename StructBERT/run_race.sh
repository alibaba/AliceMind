echo "Loading params"
training=${1:-"./RACE"}
model=${2:-"./model"}

lr=${3:-5e-5}
weight_decay=${4:-0.0}
batch_size=${5:-8}
step_size=${6:-500}
gamma=${7:-0.5}
epoch=${8:-3}

echo "Running"
CMD="race.py"
CMD+=" --model_path=$model"
CMD+=" --train_file=$training"
CMD+=" --batch_size=$batch_size"
CMD+=" --learning_rate=$lr"
CMD+=" --weight_decay=$weight_decay"
CMD+=" --gamma=$gamma"
CMD+=" --epoch=$epoch"

CMD="python3 $CMD"
echo "$CMD"

$CMD

echo "Finished Fine-tuning"
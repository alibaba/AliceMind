export CUDA_VISIBLE_DEVICES=7
export FUNSD_DIR=./funsd_dataset_structurallm
export TOKENIZER_DIR=./tokenizer_model

learning_rate=8e-6
num_train_epoch=27

CUDA_VISIBLE_DEVICES=7 python run_seq_labeling.py \
--bert_config_file=$TOKENIZER_DIR/config.json \
--init_checkpoint=./structural_lm_models/model.ckpt-160000 \
--use_position_embeddings=True \
--do_train=True \
--do_eval=True \
--train_file=$FUNSD_DIR/train \
--use_roberta=True \
--roberta_model_path=$TOKENIZER_DIR \
--do_predict=True \
--predict_file=$FUNSD_DIR/test \
--train_batch_size=2 \
--learning_rate=$learning_rate \
--labels=$FUNSD_DIR/labels.txt \
--num_train_epochs=$num_train_epoch \
--max_seq_length=448 \
--doc_stride=128 \
--output_dir=./funsd_lr${learning_rate}_epoch${num_train_epoch} \
--overwrite=True \
>> nohup_finetune_funsd_dataset_${num_train_epoch}_${learning_rate}.out 2>&1

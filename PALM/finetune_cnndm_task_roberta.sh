gpu=0,1

save_model_path=./cnndm_base_models
tokenize_model_pth=./palm_model_and_data/roberta-base/
result_path=${save_model_path}/results
data_path=./palm_model_and_data/palm_data_cnndm/cnndm
train_from=./palm_model_and_data/model_palm_en_base.pt

# train
python train.py -task abs -dataset cnn -train_from ${train_from} -dec_layers 12 -finetune_bert true -mode train -lr_dec 1e-1 -min_length 20 -max_length 140 -max_tgt_len 140 -data_path ${data_path} -model_pth ${tokenize_model_pth} -dec_dropout 0.2  -model_path ${save_model_path}  -sep_optim true -save_checkpoint_steps 4000 -batch_size 400 -train_steps 100000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 10000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus $gpu > log/nohup_finetune_cnndm.out 2>&1

# validate
python train.py -test_all true -dataset cnn -alpha 0.95 -dec_layers 12 -task abs -mode validate -data_path ${data_path}  -dec_dropout 0.2 -result_path ${result_path} -min_length 40 -max_length 130 -max_tgt_len 130 -model_path ${save_model_path} -model_pth ${tokenize_model_pth} -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -test_batch_size 2000 -batch_size 400 -train_steps 200000 -report_every 50 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  >> log/log_decode_cnndm.out 2>&1

# test
python train.py -test_from $save_model_path/model_step_68000.pt -dataset cnn -alpha 0.95 -dec_layers 12 -task abs -mode test -data_path ${data_path}  -dec_dropout 0.2 -result_path ${result_path} -min_length 40 -max_length 130 -max_tgt_len 130 -model_path ${save_model_path} -model_pth ${tokenize_model_pth} -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -test_batch_size 2000 -batch_size 400 -train_steps 200000 -report_every 50 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  >> log/log_decode_cnndm.out 2>&1

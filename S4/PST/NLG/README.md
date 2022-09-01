# Sparse GPT-2 using PST

This folder contains the implementation of PST in GPT-2 using the Python package pst and steps to replicate the results in our paper, which is based on [LoRA](https://github.com/microsoft/LoRA).

## Repository Overview
Our implementation is based on the fine-tuning code for GPT-2 in Hugging Face. There are several directories in this repo:

src/ contains the source code used for data processing, training, and decoding.

eval/ contains the code for task-specific evaluation scripts.

data/ contains the raw data we used in our experiments.

vocab/ contains the GPT-2 vocabulary files.

## Getting Started

Install dependencies and download pretrained checkpoints.

```bash
bash download_pretrained_checkpoints.sh
bash create_datasets.sh
cd ./eval
bash download_evalscript.sh
cd ..
```

### Replicating Our Result on E2E

1. Sparse GPT-2 Medium with PST (see our paper for hyperparameters for GPT-2 Medium)
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 1 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0001 \
    --weight_decay 0.0 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 5000 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 43
```

2. Generate outputs from the trained model using beam search:
```bash
python3 -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/sparse_model.pt \
    --platform local \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26289.b10p08r4.jsonl
```

3. Decode outputs from step (2)
```bash
python3 src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26289.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set
```bash
python3 eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

### Replicating Our Result on WebNLG

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with webnlg

2. Decode outputs from beam search (step 2 above)
```bash
python3 src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg/predict.20000.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower
```

3. Run evaluation on WebNLG test set
```
cd ./eval/GenerationEval/
python3 eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

### Replicating Our Result on DART

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with dart

2. Decode outputs from beam search (step 2 above)
```bash
python3 src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models/GPT2_M/dart/predict.20000.b10p08.jsonl \
        --input_file ./data/dart/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower
```

3. Run evaluation on Dart test set
```bash
cd ./eval/GenerationEval/
python3 eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```
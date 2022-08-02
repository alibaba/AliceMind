# Squad

## SQUAD V2.0
Please download all the related files from this link:
https://rajpurkar.github.io/SQuAD-explorer/

Including: train-v2.0.json, dev-v2.0.json, evaluate.py

## Guide

1. Install Transformers
```
pip install transformers
```
2. Adjust the hyperparameters in the ```run_squad.sh```.
   
   Then, run the following code:

```
bash run_squad.sh
```
3. Use the fine-tuned model to generate predictions and get exact_match and f1 scores with ```evaluate.py```.

## Example Usage

```
squad.py
  --model_path "Path to pretrained models"
  --train_file "Path to SQAUD V2.0 Trasn JSON file"
  --epoch "Number of epoches"
  --learning_rate "Learning rate for Adam"
  --weight_decay "Weight decay for training"
  --batch_size "Batch size for training"
  --step_size "Step size for learning rate scheduler"
  --gamma "Multiplicative factor of learning rate decay"
```
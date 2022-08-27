# RACE

## RACE
Please download all the related files from this link:
https://www.cs.cmu.edu/~glai1/data/race/

Including: RACE
            --train
            --dev
            --test

## Guide

1. Install Transformers(Version - 4.21.2)
```
pip install transformers
```
2. Adjust the hyperparameters in the ```run_race.sh```.
   
   Then, run the following code:

```
bash run_race.sh
```

## Example Usage

```
squad.py
  --model_path "Path to pretrained models"
  --train_file "Path to RACE file"
  --epoch "Number of epoches"
  --learning_rate "Learning rate for Adam"
  --weight_decay "Weight decay for training"
  --batch_size "Batch size for training"
  --step_size "Step size for learning rate scheduler"
  --gamma "Multiplicative factor of learning rate decay"
```

# Contrastive Pruning

Source code for AAAI 2022 paper: [From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression]().

## 1. Environments

- python==3.6.12
- cuda==10.2

## 2. Dependencies

- torch==1.8.0
- transformers==4.7.0
- datasets==1.6.0
- scikit-learn==0.24.2
- numpy==1.17.0
- scipy==1.5.4
- tensorboardX==1.8
- tqdm==4.49.0

## 3. Training and Evaluation


```bash
# Run CAP-f on QQP/MNLI/SST-2
>> bash run_glue.sh

# Run CAP-f on SQuAD v1.1
>> bash run_squad.sh
```

Note that the *TEACHER_PATH* should be configured correctly, which refers to the path of the teacher model (fine-tuned model).

You can also change the other settings in the corresponding scripts. Here are some explanation of the hyperparameters:

- CONTRASTIVE_TEMPERATURE: The temperature used in contrastive learning.
- EXTRA_EXAMPLES: The number of examples that are fetched to conduct contrastive learning.
- ALIGNREP: The way we represent the sentence, which can be either *'cls'* or *'mean-pooling'*.
- RETRAIN_EPOCH: The number of epochs of the re-training after each pruning.
- CE_LOSS_WEIGHT: The loss weight of the cross-entropy loss.
- CL_UNSUPERVISED_LOSS_WEIGHT: The loss weight of unsupervised contrastive learning.
- CL_SUPERVISED_LOSS_WEIGHT: The loss weight of supervised contrastive learning
- DISTILL_LOSS_WEIGHT: The loss weight of the distillation.
- DISTILL_TEMPERATURE: The temperature used in distillation.

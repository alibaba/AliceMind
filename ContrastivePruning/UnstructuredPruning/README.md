
# Contrastive Pruning

Source code for AAAI 2022 paper: [From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression]().

Our code is based on [Movement Pruning](https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning).

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
# Run CAP-m on QQP/MNLI
>> bash run_glue_topk_kd.sh

# Run CAP-m on SQuAD v1.1
>> bash run_squad_topk_kd.sh

# Run CAP-soft on QQP/MNLI
>> bash run_glue_soft_kd.sh

# Run CAP-soft on SQuAD v1.1
>> bash run_squad_soft_kd.sh
```

Note that the *TEACHER_PATH* should be configured correctly, which refers to the path of the teacher model (fine-tuned model).

You can also change the other settings in the corresponding scripts. Here are some explanation of the hyperparameters:

- CONTRASTIVE_TEMPERATURE: The temperature used in contrastive learning.
- EXTRA_EXAMPLES: The number of examples that are fetched to conduct contrastive learning.
- ALIGNREP: The way we represent the sentence, which can be either *'cls'* or *'mean-pooling'*.
- CL_UNSUPERVISED_LOSS_WEIGHT: The loss weight of unsupervised contrastive learning.
- CL_SUPERVISED_LOSS_WEIGHT: The loss weight of supervised contrastive learning
- TEACHER_PATH: The path of the teacher model.
- CE_LOSS_WEIGHT: The loss weight of the cross-entropy loss.
- DISTILL_LOSS_WEIGHT: The loss weight of the distillation.

Besides, note that Soft movement pruning cannot precisely determine the final sparsity ratio beforehand.
Instead, it regularize through the *FINAL_LAMBDA* hyperparameter.
Therefore, we provide some basic setting, and the precise sparsity ratio can be obtained by [counts_parameters.py](./counts_parameters.py).

- QQP
    - 90% -> *FINAL_LAMBDA*=190
    - 97% -> *FINAL_LAMBDA*=550
- MNLI
    - 90% -> *FINAL_LAMBDA*=290
    - 97% -> *FINAL_LAMBDA*=680
- SQuAD
    - 90% -> *FINAL_LAMBDA*=600
    - 97% -> *FINAL_LAMBDA*=2000

```bash
# To calculate precise sparsity ratio of the pruned model
python counts_parameters.py --pruning_method sigmoied_threshold --threshold 0.1 --serialization_dir ${YOUR_MODEL_PATH}
```



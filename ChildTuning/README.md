
# Child-Tuning

Source code for EMNLP 2021 Long paper: [Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning](https://arxiv.org/pdf/2109.05687.pdf).

## 1. Environments

- python==3.6.13
- cuda==10.2

## 2. Dependencies

- torch==1.8.0
- transformers==4.7.0
- datasets==1.6.0
- scikit-learn==0.24.2
## 3. Training and Evaluation

```bash
>> bash run.sh
```

You can change the setting in [this script](./run.sh).

## 4. Citation

If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{xu-etal-2021-childtuning,
    title = "Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning",
    author = "Runxin Xu and
    Fuli Luo and Zhiyuan Zhang and
    Chuanqi Tan and Baobao Chang and
    Songfang Huang and Fei Huang",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

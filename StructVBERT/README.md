# StructVBERT：Visual-Linguistic Pre-training for Visual Question Answering

[VQA challenge 2020 Runner up](https://drive.google.com/file/d/1zYnoQqAMpBzdVEEkrdHy6w2h4lj4pEDK/view)

## Introduction
We propose a new single-stream visual-linguistic pre-training scheme by leveraging multi-stage progressive pre-training and multi-task learning.
## Pre-trained models
|Model | Description | #params | Download |
|------------------------|-------------------------------------------|------|------|
|structvbert.en.base | StructVBERT using the BERT-base architecture | 110M | [structvbert.en.base](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/pretrained_model.tar.gz) |
|structvbert.en.large | StructVBERT using the BERT-large architecture | 355M | Coming soon |

## Results
The results of VQA & NLVR2 tasks can be reproduced using the hyperparameters listed in the following "Example usage" section.
#### structvbert.en.base

| Split | [VQA](https://visualqa.org/) | [NLVR2](https://lil.nlp.cornell.edu/nlvr/) |
|--------------------|-------|-------|
|  Local Validation	 |71.80% |77.66% |
| Test-Dev |74.11% |78.13% (Test-P) |

## Example usage
#### Requirements and Installation
* [PyTorch](https://pytorch.org/) version >= 1.3.0

* Install other libraries via
```
pip install -r requirements.txt
```

* For faster training install NVIDIA's [apex](https://github.com/NVIDIA/apex) library
* The codebase is built on top of [LXMERT](https://github.com/airsplay/lxmert) codebase. Please first download the [pretrained structvbert model](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/pretrained_model.tar.gz), [VQA and NLVR2 data](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/data.tar.gz) and VQA and NLVR2 image features [VQA train](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/train_npz.tar.gz), [VQA val](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/valid_npz.tar.gz), [VQA test](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/test_npz.tar.gz), [NLVR2 train](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/nlvr2_train_npz.tar.gz), [NLVR2 val](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/nlvr2_valid_npz.tar.gz) and [NLVR2 test](http://119608.oss-cn-hangzhou-zmf.aliyuncs.com/structvbert/nlvr2_test_npz.tar.gz). 

* After downloading the data and features from the drives, 
please re-organize them according to the following example:
```
REPO ROOT
 |
 |-- data                  
 |    |-- vqa
 |    |    |-- train.json
 |    |    |-- minival.json
 |    |    |-- nominival.json
 |    |    |-- test.json
 |    |    |-- trainval_ans2label.json
 |    |    |-- trainval_label2ans.json
 |    |    |-- all_ans.json
 |    |    |-- coco_minival_img_ids.json
 |    |
 |    |-- mscoco_imgfeat
 |    |    |-- train_npz
 |    |    |    |-- *.npz
 |    |    |-- valid_npz
 |    |    |    |-- *.npz
 |    |    |-- test_npz
 |    |	   |    |-- *.npz
 |    |
 |    |-- nlvr2_imgfeat
 |    |    |-- nlvr2_train_npz
 |    |    |    |-- *.npz
 |    |    |-- nlvr2_valid_npz
 |    |    |    |-- *.npz
 |    |    |-- nlvr2_test_npz
 |    |    |    |-- *.npz
 | 
 |-- pretrained_model
 |    |-- bert_config.json
 |    |-- pytorch_model.bin
 |    |-- vocab.txt
 |-- lxrt
 |-- tasks
 |-- run_vqa.sh
 |-- run_vqa_predict.sh
 |-- run_nlvr.sh
 |-- *.py
```
Please also kindly contact us if anything is missing!

####  VQA
##### Fine-tuning
After all the data and models are downloaded and arranged as the example of  REPO ROOT, you can directly finetune with our script as:
```
sh run_vqa.sh
```
Now we use NVIDIA's [apex](https://github.com/NVIDIA/apex) library for faster training, which can increase the speed by more than 1.5 times. 
**Note**: it can only be used on V100 machine, if you do not have V100 GPU card, please remove the **--amp_type O1** option.

##### Predict & Submitted to VQA test server
```
sh run_vqa_predict.sh
```
The test results will be saved in `output/$name/test_predict.json`. The VQA 2.0 challenge for this year is host on [EvalAI](https://evalai.cloudcv.org/) at [https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278). After registration, you should only upload the `test_predict.json` file to the server and check the submission result. 

####  NLVR2
##### Fine-tuning
After all the data and models are downloaded and arranged as the example of  REPO ROOT, just as fine-tuning for VQA, you can directly finetune with our script as:
For local validation set, 
```
sh run_nlvr_val.sh
```
For Test-P set,
```
sh run_nlvr_test.sh
```

## Reference
For more details, you can also see our [slides](https://drive.google.com/file/d/1zYnoQqAMpBzdVEEkrdHy6w2h4lj4pEDK/view) and [talk](https://www.youtube.com/watch?v=XvLi-QQh7wk) on CVPR 2020 VQA Challenge Workshop.

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_vqa_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_sampler, create_loader
from dataset.videoqa_dataset import videoqa_dataset


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        # topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
        B, T, C, H, W = image.shape
        image = image.view(-1, C, H, W)
        image_embeds = model.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=model.use_checkpoint)
        if model.large:
            image_embeds = model.dropout(model.visn_layer_norm(model.visn_fc(image_embeds)))
        image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text_output = model.text_encoder(question_input.input_ids, attention_mask=question_input.attention_mask,
                                         return_dict=True)
        text_embeds = text_output.last_hidden_state
        fusion_output = model.fusion_encoder(encoder_embeds=text_embeds,
                                             attention_mask=question_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=False)
        image_output, question_output = fusion_output
        question_output = torch.cat([image_output, question_output], 1)
        merge_text_attention = torch.cat([image_atts, question_input.attention_mask], 1)
        topk_ids, topk_probs = model.rank_answer(question_output, merge_text_attention, answer_input.input_ids,
                                                 answer_input.attention_mask, config['k_test'])

        result = []
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
        accuracy = cal_metric(result, dataset)

        metric_logger.meters['acc'].update(accuracy, n=B)

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(vqa_result, val_file):
    with open(val_file[0], "r") as f:
        data_list = [json.loads(l.strip("\n")) for l in f.readlines()]
    id2datum = {}
    for idx, each in enumerate(data_list):
        question_id = idx
        id2datum[question_id] = {
            'question': each['question'],
            'video_id': each['video_id'],
            'answer': each['answer'],
        }
    score = 0.
    for each in vqa_result:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]['answer']
        if label == ans:
            score += 1
    return score / len(vqa_result)


def main(args, config):
    print('master addr: ', os.environ['MASTER_ADDR'])
    print('master port: ', os.environ['MASTER_PORT'])
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change

        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                               pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        for key in list(state_dict.keys()):
            if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                encoder_key = key.replace('fusion.', '').replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    device = torch.device(args.device + ':' + os.environ['LOCAL_RANK'])
    print('local device:', device)

    model = model.to(device)

    #### Dataset ####
    print("Creating videoqa datasets")
    datasets = [videoqa_dataset(
        config['val_file'],
        None,
        config['videoqa_root'],
        split='test',
        answer_list=config['answer_list'],
        read_local_data=config['read_local_data'],
        max_img_size=config['image_res']
    )]

    if "msvd" in config['val_file'][0]:
        datasets[0].video_fmt = '.avi'
    print(datasets[0].video_fmt)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [False], num_tasks, global_rank)
    else:
        samplers = [None]

    val_loader = \
    create_loader(datasets, samplers, batch_size=[config['batch_size_test']], num_workers=[8], is_trains=[False],
                  collate_fns=[None])[0]

    print("Start Evaluate")
    start_time = time.time()

    val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config)
    print(val_stats)

    if utils.is_main_process():
        log_stats = {**{f'{k}': v for k, v in val_stats.items()}, }
        with open(os.path.join(args.output_dir, "test_result.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
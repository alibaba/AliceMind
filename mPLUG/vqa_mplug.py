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
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

import deepspeed

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=args.max_input_length if config["add_ocr"] else 25, return_tensors="pt").to(
            device)
        if i == 0:
            print ("question: ", question)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        #if accum_steps > 1:
        #    loss = loss / accum_steps

        #if do_amp:
        #    from apex import amp
        #    with amp.scale_loss(loss, optimizer) as scaled_loss:
        #        # logger.info('scaled loss: {}'.format(str(scaled_loss)))
        #        scaled_loss.backward()
        #else:
        #    loss.backward()
        #if (i + 1) % accum_steps == 0:
        #    optimizer.step()
        #    optimizer.zero_grad()
        model.backward(loss)
        model.step()
        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        del image,weights, question_input,answer_input, loss
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id":ques_id, "answer":ans})   

    return result

@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        result = []
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id":ques_id, "answer":ans})   
        accuracy = cal_metric(result, dataset)
        # accuracy = (targets == pred_class).sum() / targets.size(0)
        #
        metric_logger.meters['acc'].update(accuracy, n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
def cal_metric(vqa_result, val_file):
    
    with open(val_file[0], "r") as f:
        data_list = json.load(f)
    id2datum = {}
    for each in data_list:
        id2datum[each["question_id"]] = each["label"]
    score = 0.
    for each in vqa_result:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]
        if ans in label:
            score += label[ans]
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

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change

        if not args.evaluate:
            if config["clip_name"] == "ViT-B-16":
                num_patches = int(config["image_res"] * config["image_res"]/(16*16))
            elif config["clip_name"] == "ViT-L-14":
                num_patches = int(config["image_res"] * config["image_res"]/(14*14))
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

            pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
            state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
            if config['distill']:
                if config["clip_name"] == "ViT-B-16":
                    num_patches = int(config["image_res"] * config["image_res"]/(16*16))
                elif config["clip_name"] == "ViT-L-14":
                    num_patches = int(config["image_res"] * config["image_res"]/(14*14))
                pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

                pos_embed = resize_pos_embed(state_dict['visual_encoder_m.visual.positional_embedding'].unsqueeze(0),
                                             pos_embed.unsqueeze(0))
                state_dict['visual_encoder_m.visual.positional_embedding'] = pos_embed

            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    if args.deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            dist_init_required=True
        )

    device = torch.device(args.device + ':' + os.environ['LOCAL_RANK'])
    print('local device:', device)

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[12,8,8],is_trains=[True, False, False],
                                              collate_fns=[vqa_collate_fn,None,None])



    print("Start training")
    start_time = time.time()

    val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config)
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)
            model.save_checkpoint(os.path.join(args.output_dir), tag='{}.pt'.format(model.global_steps))
        val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config)
        if epoch >= 5:
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)
            result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

        if args.evaluate:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


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
    parser.add_argument('--max_input_length', default=50, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--add_ocr', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--accum_steps', default=1, type=int)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['add_ocr'] = args.add_ocr
    config['add_object'] = args.add_object
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

import argparse
import os
import ruamel_yaml as yaml
import language_evaluation
from torch.autograd import Variable
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

from models.model_caption_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, coco_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

import language_evaluation.coco_caption_py3.pycocoevalcap as evaluation_tools
import multiprocessing
import itertools


def train_scst(model, data_loader, test_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    beam_size=args.beam_size
    tokenizer_pool=multiprocessing.Pool()
    best_cider = 0.0
    for i, (image, caption, object_labels, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question_input, caption, train=True,out_size=beam_size,scst=True)

        probs_list=[]
        for item in topk_probs:
            probs_list.append(torch.stack(item,dim=0))
        topk_probs_tensor=torch.stack(probs_list,dim=0)

        caps_gen=[]
        topk_words=[]
        for img_item in topk_ids:
            words=[]
            for item in img_item:
                caps_gen.append(tokenizer.decode(item).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip())
                words.append(item.numel())
            topk_words.append(words)
        topk_words_tensor = torch.Tensor(topk_words).cuda()
        caps_gt = gold_caption
        caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
        caps_gen,caps_gt = tokenizer_pool.map(evaluation_tools.PTBTokenizer.tokenize,[caps_gen,caps_gt])

        reward=evaluation_tools.compute_ciders(caps_gt,caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).cuda().view(image.shape[0], beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = - (topk_probs_tensor/topk_words_tensor) * (reward-reward_baseline)
        loss = loss.mean()
        #loss.requires_grad_(True)

        #loss = Variable(loss, requires_grad = True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        del image, question_input,caption,loss 
        if i > 0 and i % args.eval_steps == 0:
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)
            result_file = save_result(vqa_result, args.result_dir, 'vqa_result_%d' % i)
            model.eval()
            if utils.is_main_process():
                result = cal_metric(result_file)
                print('*'*100)
                print(type(result))
                print(result)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps({'Starting_Training':result}) + "\n")
                if result["CIDEr"]*100 > best_cider:
                    best_cider = result["CIDEr"]*100

                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': i,
                    }, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            dist.barrier()
            model.train()


            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


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
    for i, (image, caption, object_labels, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        if config['prompt'] != "":
            caption = [config['prompt'] + each+config['eos'] for each in caption]
        else:
            caption = [each+config['eos'] for each in caption]

        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        # question_input = caption.input_ids[0,0].repeat(caption.input_ids.size(0), 1)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, caption, train=True)
        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        del image, question_input,caption,loss 

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, test_submit=False):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []


    answer_input = None
    for n, (image, caption, object_labels, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question_input, caption, train=False)

        for image_id, topk_id, topk_prob, gold_caption_list in zip(image_ids, topk_ids, topk_probs, gold_caption):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            if test_submit:
                result.append({"image_id": int(image_id.replace(".jpg", "").split("_")[-1]), "caption":ans})   
            else:
                result.append({"question_id":image_id, "pred_caption":ans, "gold_caption":gold_caption_list})   
    return result

@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    predicts = []
    answers = []
    answer_input = None
    for n, (image, caption, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])
        print (predicts, answers)
        #{'Bleu_1': 0.9999999999863945, 'Bleu_2': 0.9999999999859791, 'Bleu_3': 0.9999999999854866, 'Bleu_4': 0.999999999984889, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 2.7246232035629268, 'SPICE': 0.40389416048620613}
        result = cal_metric(predicts, answers)
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_2'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_3'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_4'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print (len(result_list), results)
    return results

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('coco', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[8,8,8],is_trains=[True, False, False], 
                                              collate_fns=[coco_collate_fn, coco_collate_fn, coco_collate_fn]) 


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            #state_dict = checkpoint['model']
            state_dict = checkpoint['model']
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict[key]
        except:
            state_dict = checkpoint['module']

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

            
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_scst(model, train_loader, test_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        if args.evaluate:
            break

        vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)
        if utils.is_main_process():
            result = cal_metric(result_file)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps({'Epoch['+str(epoch)+']':result}) + "\n")
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

        dist.barrier()
        
    #vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    #result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

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
    parser.add_argument('--eval_steps', default=1500, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--open_generation', action='store_true')
    parser.add_argument('--merge_attention', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--concat_last_layer', action='store_true')
    parser.add_argument('--add_ocr', action='store_true')
    parser.add_argument('--has_decode', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--decode_layers', default=12, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['add_object'] = args.add_object
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder


    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

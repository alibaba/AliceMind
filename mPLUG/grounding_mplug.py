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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils as public_utils
from torch.utils.data import DataLoader
from dataset.grounding_dataset import NestedTensor, collate_fn, collate_fn_val

from models.model_grounding_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

from vgTools.utils import misc as utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer
from vgTools.utils import eval_utils
from icecream import ic
from pdb import set_trace as breakpoint

def load_checkpoint(model,checkpoint_path,args,config):
    if isinstance(model,torch.nn.parallel.DistributedDataParallel):
        model=model.module
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    tmp = {}
    for key in state_dict.keys():
        if '_m.' in key:
            continue
        if 'text_encoder.bert' in key[:len('text_encoder.bert')]:
            encoder_key = key.replace('bert.', '')
            tmp[encoder_key] = state_dict[key]
        elif 'fusion_encoder.fusion' in key:
            encoder_key = key.replace('fusion.', '')
            tmp[encoder_key]=state_dict[key]
        else:
            tmp[key]=state_dict[key]

    state_dict = tmp

    # reshape positional embedding to accomodate for image resolution change
    vit_rate = 16*16 if '16' in config['clip_name'] else 14*14
    num_patches = int(config["image_res"] * config["image_res"]/vit_rate)
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, config['vision_width']).float())

    pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                pos_embed.unsqueeze(0))
    state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

    if not args.evaluate:
        if config['distill']:
            num_patches = int(config["image_res"] * config["image_res"] / vit_rate)
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, config['vision_width']).float())

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % checkpoint_path)
    print(msg)

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_two_optim=False,do_amp=False):
    accum_steps=config.get('accum_steps',1)
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    
    metric_logger.add_meter('loss_seq', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
  
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
        loss_dict = model(img_data, text_data,{'targets':target})
        loss = sum(loss_dict[k] for k in loss_dict.keys())
        
        optimizer.zero_grad()
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
        metric_logger.update(loss_seq=loss_dict['loss_seq'].item())
        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val(model, data_loader, tokenizer, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target,raw_data = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        
        pred_res = model(img_data, text_data,{})
    
        pred_boxes=pred_res
        
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)
        
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    pred_box_list = []
    gt_box_list = []

    from tqdm import tqdm
    
    for _, batch in enumerate(tqdm(data_loader)):

        img_data, text_data, target,raw_data = batch
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        pred_res = model.module(img_data, text_data,{})
        pred_boxes=pred_res

        pred_box_list.append(pred_boxes.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)
    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    return accuracy

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
    print("Creating dataset")
    train_dataset, val_dataset, test_datasets = create_dataset(config['dataset'], config) 
    datasets = [train_dataset, val_dataset]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)              
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size_train'],config['batch_size_train']],
                                              num_workers=[48,48],is_trains=[True, False], collate_fns=[collate_fn,collate_fn_val])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    #### Model ####
    print("Creating model")
    model = MPLUG(config = config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)

    for name, module in model.named_modules():
        if hasattr(module,'use_checkpoint') and module.use_checkpoint==True:
            module.use_checkpoint=False
            print(f"Set {name} checkpointing: False")
        if hasattr(module,'config') and getattr(module.config, "gradient_checkpointing", False):
            module.config.gradient_checkpointing=False
            print(f"Set {name} checkpointing: False")

    if args.do_two_optim:
        arg_opt = public_utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)
    else:
        arg_opt = public_utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)

    arg_sche = public_utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        load_checkpoint(model,args.checkpoint,args,config)
    
 
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if int(torch.__version__.split('.')[1])>=10:
            model._set_static_graph()
        model_without_ddp = model.module
    if not args.evaluate:
        print("Start training")
        start_time = time.time()
        best_accu = 0
        for epoch in range(start_epoch, max_epoch):
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)

            if not args.evaluate:
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)

                train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                    config, do_amp=args.do_amp, do_two_optim=args.do_two_optim)
    
            results = val(model, test_loader, tokenizer, device)


            if utils.is_main_process():              
                if args.evaluate:      
                    log_stats = {**{f'{k}': v for k, v in results.items()},
                                'epoch': epoch,
                                }                   
                else:             
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'{k}': v for k, v in results.items()},
                                'epoch': epoch,
                                }      
                    if results['accu']>best_accu:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                        best_accu = results['accu']
                    if (epoch + 1) % 10 == 0:
                        checkpoint_path=(Path(args.output_dir , f'checkpoint{epoch:04}.pth'))
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'val_accu': results['accu']
                        }, checkpoint_path)
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n") 
                    
            if args.evaluate: 
                break                
            
            lr_scheduler.step(epoch+warmup_steps+1)  
            dist.barrier()   

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
    # Eval
    from torch.utils.data import DataLoader, DistributedSampler
    checkpoint_path=''
    if Path(args.eval_checkpoint).exists():
        checkpoint_path=args.eval_checkpoint
        load_checkpoint(model,args.eval_checkpoint,args,config)
    else:
        print(f'checkpoint {args.eval_checkpoint} not found.')
        if Path(args.output_dir,'checkpoint_best.pth').exists():
            checkpoint_path=Path(args.output_dir,'checkpoint_best.pth')
            print(f'load default best checkpoint')
            load_checkpoint(model,Path(args.output_dir,'checkpoint_best.pth'),args,config)
        else:
            print('no checkpoint available.')
            import sys
            sys.exit(0)

    for split_name,split_dataset in test_datasets.items():
        if args.distributed:
            sampler_test = DistributedSampler(split_dataset)
        else:
            sampler_test = torch.utils.data.SequentialSampler(split_dataset)

        data_loader_test = DataLoader(split_dataset, 1, sampler=sampler_test,
                                    drop_last=False, collate_fn=collate_fn_val, num_workers=12)
        start_time = time.time()
        accuracy = evaluate(model,data_loader_test,tokenizer,device)
        if utils.is_main_process():
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

            log_stats = {'test_model:': str(checkpoint_path),
                        '%s_set_accuracy'%split_name: accuracy,
                        }
            print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (Path(args.output_dir) / "eval_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--eval_checkpoint', default='')
    parser.add_argument('--output_dir', default='output/RefCOCO')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--dataset', default='vg_uni')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    if args.finetune:
        config['optimizer']['lr1']=2e-6
        config['optimizer']['lr2']=2e-6
    if 'clip_name' not in config:
        config['clip_name'] = 'ViT-B-16.tar'
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)


    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

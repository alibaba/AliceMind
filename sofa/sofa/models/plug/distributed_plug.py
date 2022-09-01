import torch
import deepspeed

from sofa.learning_rates import AnnealingLR
from sofa.utils import print_rank_0, mpu,load_deepspeed_checkpoint
from sofa.utils.fp16 import FP16_Module, FP16_Optimizer
from apex.optimizers import FusedAdam as Adam
from .modeling_plug import BertLayerNorm
from sofa.deepspeed_initializer import initialize_distributed, set_random_seed

USE_TORCH_DDP = False
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from sofa.distributed import DistributedDataParallel as DDP

class DistributedPlug(object):
    def __init__(self, tokenizer, args):
        super().__init__()
        torch.backends.cudnn.enabled = False
        self.tokenizer = tokenizer
        self.args = args
        initialize_distributed(args)
        set_random_seed(args.seed)
    
    def set_model(self, model):
        self.model = model
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def get_model(self, args):
        """Build the model."""

        print_rank_0('building BERT model ...')
        model = self.model

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)

        if args.deepspeed and args.fp16: 
            model.half()   

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if args.fp16:
            model = FP16_Module(model)
            if args.fp32_embedding:
                model.module.model.bert.embeddings.word_embeddings.float()
                model.module.model.bert.embeddings.position_embeddings.float()
                model.module.model.bert.embeddings.token_type_embeddings.float()
            if args.fp32_tokentypes:
                model.module.model.bert.embeddings.token_type_embeddings.float()
            if args.fp32_layernorm:
                for name, _module in model.named_modules():
                    if 'LayerNorm' in name:
                        _module.float()

        # Wrap model for distributed training.
        if USE_TORCH_DDP:
            i = torch.cuda.current_device()
            model = DDP(model, device_ids=[i], output_device=i,
                        process_group=mpu.get_data_parallel_group())
        else:
            model = DDP(model)

        return model

    def setup_tokenizer_stuff(self, args):
        if mpu.get_model_parallel_rank() == 0:
            args.do_train = True
            args.do_valid = True
            args.do_test = True
            tokenizer = self.tokenizer
            before = tokenizer.num_tokens
            after = before
            multiple = args.make_vocab_size_divisible_by / 2 * \
                    mpu.get_model_parallel_world_size()
            while (after % multiple) != 0:
                after += 1
            print_rank_0('> padded vocab (size: {}) with {} dummy '
                        'tokens (new size: {})'.format(
                            before, after - before, after))
            # Need to broadcast num_tokens and num_type_tokens.
            token_counts = torch.cuda.LongTensor([after,
                                                tokenizer.num_type_tokens,
                                                int(args.do_train), int(args.do_valid), int(args.do_test)])
        else:
            tokenizer = None
            token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

        # Broadcast num tokens.
        torch.distributed.broadcast(token_counts,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        num_tokens = token_counts[0].item()
        num_type_tokens = token_counts[1].item()
        args.do_train = token_counts[2].item()
        args.do_valid = token_counts[3].item()
        args.do_test = token_counts[4].item()
        return tokenizer, num_tokens, num_type_tokens

    def get_params_for_weight_decay_optimization(self, module):

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        
        for module_ in module.modules():
            if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                    if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and 'mask_score' not in n and 'mask' not in n and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'])

        return weight_decay_params, no_weight_decay_params

    def get_optimizer(self, model, args):
        """Set up the optimizer."""

        # Build parameter groups (weight decay and non-decay).
        while isinstance(model, (DDP, FP16_Module)):
            model = model.module
        layers = model.model.bert.encoder.layer
        pooler = model.model.bert.pooler
        #lmheads = model.model.cls.predictions
        nspheads = model.model.cls
        embeddings = model.model.bert.embeddings
        param_groups = []
        param_groups += list(self.get_params_for_weight_decay_optimization(layers))
        param_groups += list(self.get_params_for_weight_decay_optimization(pooler))
        param_groups += list(self.get_params_for_weight_decay_optimization(nspheads))
        param_groups += list(self.get_params_for_weight_decay_optimization(embeddings))
        #param_groups += list(get_params_for_weight_decay_optimization(
        #    lmheads.transform))
        #param_groups[1]['params'].append(lmheads.bias)

        # Add model parallel attribute if it is not set.
        for param_group in param_groups:
            for param in param_group['params']:
                if not hasattr(param, 'model_parallel'):
                    param.model_parallel = False

        if args.cpu_optimizer:
            if args.cpu_torch_adam:
                cpu_adam_optimizer = torch.optim.Adam 
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam 
                cpu_adam_optimizer = DeepSpeedCPUAdam
            optimizer = cpu_adam_optimizer(param_groups,
                            lr=args.lr, weight_decay=args.weight_decay)
        else:
            # Use Adam.
            optimizer = Adam(param_groups,
                            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
            #optimizer = torch.optim.SGD(param_groups, lr=args.lr * 10)
        print(f'Optimizer = {optimizer.__class__.__name__}') 
        if args.deepspeed:
            # fp16 wrapper is not required for DeepSpeed. 
            return optimizer

        # Wrap into fp16 optimizer.
        if args.fp16:
            optimizer = FP16_Optimizer(optimizer,
                                    static_loss_scale=args.loss_scale,
                                    dynamic_loss_scale=args.dynamic_loss_scale,
                                    dynamic_loss_args={
                                        'scale_window': args.loss_scale_window,
                                        'min_scale':args.min_scale,
                                        'delayed_shift': args.hysteresis})

        return optimizer

    def get_learning_rate_scheduler(self, optimizer, args):
        """Build the learning rate scheduler."""

        # Add linear learning rate scheduler.
        if args.lr_decay_iters is not None:
            num_iters = args.lr_decay_iters
        else:
            num_iters = args.train_iters
        init_step = -1
        warmup_iter = args.warmup * num_iters
        #print ("####: ", warmup_iter, args.warmup, num_iters)
        lr_scheduler = AnnealingLR(optimizer,
                                start_lr=args.lr,
                                warmup_iter=warmup_iter,
                                num_iters=num_iters,
                                decay_style=args.lr_decay_style,
                                last_iter=init_step)

        return lr_scheduler

    def setup_model_and_optimizer(self):
        """Setup model and optimizer."""
        args = self.args
        
        model = self.get_model(args)
        if args.pre_load:
            from sofa.utils import pre_load
            load_model = pre_load(mpu, args.load, args.load_iteration)
            model_dict = model.module.module.model.state_dict() 
            for key in load_model:
                if key not in model_dict.keys():
                    print_rank_0('Skip key: '+key)
                else:
                    print_rank_0('Loading key: '+key)
            model.module.module.model.load_state_dict(pre_load(mpu, args.load, args.load_iteration), strict=False)
            args.iteration = 0
        optimizer = self.get_optimizer(model, args)
        lr_scheduler = self.get_learning_rate_scheduler(optimizer, args)

        pruner = None
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")

            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=False
        )
            model.save_zero_checkpoint = False
        if not args.pre_load:
            if args.load is not None:
                args.iteration = load_deepspeed_checkpoint(model, optimizer, lr_scheduler, args)
            else:
                args.iteration = 0
        return model, optimizer, lr_scheduler, pruner


class DistributedPlugNLG(DistributedPlug):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        
    def get_optimizer(self, model, args):
        """Set up the optimizer."""
        
        # Build parameter groups (weight decay and non-decay).
        while isinstance(model, (DDP, FP16_Module)):
            model = model.module
        layers = model.model.bert.encoder.layer
        embeddings = model.model.bert.embeddings
        dec_layers = model.model.decoder.decoder
        #dec_embeddings = model.model.decoder.embeddings
        #layer_norm = model.model.decoder.decoder.final_layernorm
        param_groups = []
        param_groups += list(self.get_params_for_weight_decay_optimization(layers))
        param_groups += list(self.get_params_for_weight_decay_optimization(embeddings))
        param_groups += list(self.get_params_for_weight_decay_optimization(dec_layers))
        #param_groups += list(get_params_for_weight_decay_optimization(dec_embeddings))
        #param_groups += list(get_params_for_weight_decay_optimization(layer_norm))

        # Add model parallel attribute if it is not set.
        for param_group in param_groups:
            for param in param_group['params']:
                if not hasattr(param, 'model_parallel'):
                    param.model_parallel = False

        if args.cpu_optimizer:
            if args.cpu_torch_adam:
                cpu_adam_optimizer = torch.optim.Adam 
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam 
                cpu_adam_optimizer = DeepSpeedCPUAdam
            optimizer = cpu_adam_optimizer(param_groups,
                            lr=args.lr, weight_decay=args.weight_decay)
        else:
            # Use Adam.
            optimizer = Adam(param_groups,
                            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-8)

        print(f'Optimizer = {optimizer.__class__.__name__}') 
        if args.deepspeed:
            # fp16 wrapper is not required for DeepSpeed. 
            return optimizer

        # Wrap into fp16 optimizer.
        if args.fp16:
            optimizer = FP16_Optimizer(optimizer,
                                    static_loss_scale=args.loss_scale,
                                    dynamic_loss_scale=args.dynamic_loss_scale,
                                    dynamic_loss_args={
                                        'scale_window': args.loss_scale_window,
                                        'min_scale':args.min_scale,
                                        'delayed_shift': args.hysteresis})

        return optimizer

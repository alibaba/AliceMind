import os
import random
import numpy as np
import torch

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from sofa.utils import mpu,print_rank_0

def load_checkpoint(model,
                    load_dir,
                    tag,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True):
    r"""Load training checkpoint

    Arguments:
        load_dir: Required. Directory to load the checkpoint from
        tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.
        load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
        load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
        load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
    Return:
        load_path: Path of the loaded checkpoint. None if loading the checkpoint failed
        client_state: State dictionary used for loading required training states in the client code.
    """

    load_path, client_states = _load_checkpoint(model,
                                                load_dir,
                                                tag,
                                                load_module_strict=load_module_strict,
                                                load_optimizer_states=load_optimizer_states,
                                                load_lr_scheduler_states=load_lr_scheduler_states)

    if load_optimizer_states:
        if model.zero_optimization() and load_path is not None:
            model._load_zero_checkpoint(load_dir,
                                       tag,
                                       load_optimizer_states=load_optimizer_states)

    return load_path, client_states

def _get_ckpt_name(mpu, checkpoints_path, tag):
    mp_rank = 0 if mpu is None else mpu.get_model_parallel_rank()
    ckpt_name = os.path.join(checkpoints_path,
                             str(tag),
                             'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')
    return ckpt_name

def pre_load(mpu,
             load_dir,
             tag):
    load_path = _get_ckpt_name(mpu, load_dir, tag)
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    return checkpoint['module']

def _load_checkpoint(model,
                     load_dir,
                     tag,
                     load_module_strict=True,
                     load_optimizer_states=True,
                     load_lr_scheduler_states=True):

    load_path = model._get_ckpt_name(load_dir, tag)

    if not os.path.exists(load_path):
        return None, None

    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)

    model.load_module_state_dict(state_dict=checkpoint['module'],
                                strict=load_module_strict)
    if not model.zero_optimization() and load_optimizer_states:
        if model.fp16_enabled():
            model.optimizer.load_state_dict(
                checkpoint['optimizer'],
                load_optimizer_states=load_optimizer_states)
        elif load_optimizer_states:
            model.optimizer.load_state_dict(checkpoint['optimizer'])

    if load_lr_scheduler_states and model.lr_scheduler is not None:
        model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.csr_tensor_module_names = checkpoint['csr_tensor_module_names']
    model.global_steps = checkpoint['global_steps']
    model.global_samples = checkpoint.get('global_samples',
                                         model.global_steps * model.train_batch_size())
    model.skipped_steps = checkpoint['skipped_steps']
    model.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
    model.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']
    deepspeed_states = [
        'module',
        'optimizer',
        'lr_scheduler',
        'csr_tensor_module_names',
        'skipped_steps',
        'global_steps',
        'dp_world_size',
        'mp_world_size'
    ]
    client_state = {
        key: value
        for key,
        value in checkpoint.items() if not key in deepspeed_states
    }

    return load_path, client_state

def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = 'iter_{:07d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(mpu.get_model_parallel_rank()),
                        'model_optim_rng.pt')

def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')

def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))

def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    if args.deepspeed:
        save_ds_checkpoint(iteration, model, args)
    else:
        # Only rank zer0 of the data parallel writes to the disk.
        if isinstance(model, torchDDP):
            model = model.module

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration, checkpoint_name))

            sd = {}
            sd['iteration'] = iteration
            sd['model'] = model.state_dict()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()


            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()

def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
        
    model.save_checkpoint(args.save, iteration, client_state = sd)

def get_checkpoint_iteration(args):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)
    
    return iteration, release, True

def load_deepspeed_checkpoint(model, optimizer, lr_scheduler, args):
    """Load a model checkpoint."""

    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0
        
    if args.deepspeed:

        #checkpoint_name, sd = model.load_checkpoint(args.load, iteration, load_optimizer_states=False, load_lr_scheduler_states=False)
        checkpoint_name, sd = load_checkpoint(model, args.load, iteration, load_optimizer_states=not args.no_load_optim, load_lr_scheduler_states=not args.no_load_lr)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return iteration

    else:
        
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.load, iteration, release)
        
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        if isinstance(model, torchDDP):
            model = model.module
        
        # Model.
        try:
            model.load_state_dict(sd['model'])
        except KeyError:
            print_rank_0('A metadata file exists but unable to load model '
                        'from checkpoint {}, exiting'.format(checkpoint_name))
            exit()

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer '
                            'state.'.format(checkpoint_name))
                exit()

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try: # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                            ' from checkpoint {}, exiting'.format(checkpoint_name))
                exit()
                
    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                        'Specify --no-load-optim or --finetune to prevent '
                        'attempting to load the optimizer '
                        'state.'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration

def load_weights(src, dst, dst2src=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt2model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in  str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)
#        dst._parameters[n].data.copy_(data)

def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)

def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)

def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)

def move_weights(our, oai, dst2src=False):
    """
    Loads weights from `oai` to `our` via in place copy.
    `oai` is a huggingface gpt2model, while `our` is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src=True is still untested
    """
#    while isinstance(our, (torchDDP, model.distributed.DistributedDataParallel, FP16_Module)):
#        our=our.module
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm, dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)

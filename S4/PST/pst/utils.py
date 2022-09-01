import os
import torch
import torch.nn as nn

from collections import OrderedDict

from pst.sparse import SparseLinear

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def convert_sparse_network(
    model,
    pruning_method,
    weight_rank,
    weight_beta,
    mask_rank,
    mask_alpha1,
    mask_alpha2,
    logger=None
):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            new_module = SparseLinear(module.in_features, module.out_features,
                module.bias is not None, pruning_method, weight_rank, weight_beta,
                mask_rank, mask_alpha1, mask_alpha2)

            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            
            # replace original module by new sparse module
            _setattr(model, name, new_module)

            if logger:
                logger.info(f"convert {name} to sparse.")
            else:
                print(f"convert {name} to sparse.")

def update_network_sparsity(model, sparsity):
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.cur_sparsity = sparsity

def schedule_sparsity_ratio(
    step,
    total_step,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (mul_coeff ** 3)
    return sparsity

def save_sparse_model(model, save_path, logger=None):
    # convert dense weight to sparse weight
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.convert()

    # remove unnecessary params
    model_state_dict = model.state_dict()
    save_state_dict = OrderedDict()
    for key in model_state_dict:
        if 'mask_scores' not in key and 'weight_U' not in key and 'weight_V' not in key:
            save_state_dict[key] = model_state_dict[key]

    torch.save(save_state_dict, save_path)

    # convert sparse weight to original weight
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.restore()
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from itertools import islice
from tqdm import tqdm
from math import sqrt
from collections import defaultdict


# borrow some code from https://github.com/pmichel31415/pytorch-pretrained-BERT/blob/paul/examples/pruning.py
# and https://github.com/pmichel31415/pytorch-pretrained-BERT/blob/paul/examples/classifier_eval.py
# and https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/DynaBERT/run_glue.py

def determine_pruning_sequence(
    prune_percents,
    n_heads,
    n_layers,
    n_intermediate,
    at_least_x_heads_per_layer=1,
):
    '''
    Same ratio for attention heads and MLPs
    '''

    # Compute the number of heads to prune on percentage if needed
    all_n_to_prune = []
    for prune_percent in prune_percents:
        total_heads = n_heads * n_layers
        n_to_prune = int(total_heads * prune_percent / 100)
        # Make sure we keep at least one head per layer
        if at_least_x_heads_per_layer > 0:
            if n_to_prune > total_heads - at_least_x_heads_per_layer * n_layers:
                assert False
        all_n_to_prune.append(n_to_prune)

    # We'll incrementally prune layers and evaluate
    all_n_to_prune = sorted(all_n_to_prune)
    n_to_prune_sequence_head = all_n_to_prune[:]
    for idx in range(1, len(all_n_to_prune)):
        n_to_prune_sequence_head[idx] = all_n_to_prune[idx] - all_n_to_prune[idx-1]
    # Verify that the total number of heads pruned stayed the same
    assert all_n_to_prune[-1] == sum(n_to_prune_sequence_head)

    # MLP
    all_n_to_prune = []
    for prune_percent in prune_percents:
        total_intermediate = n_layers * n_intermediate
        n_to_prune = int(total_intermediate * prune_percent / 100)
        all_n_to_prune.append(n_to_prune)
    n_to_prune_sequence_intermediate  = [0 for _ in range(len(all_n_to_prune))]
    n_to_prune_sequence_intermediate[0] = all_n_to_prune[0]
    for idx in range(1, len(all_n_to_prune)):
        n_to_prune_sequence_intermediate[idx] = all_n_to_prune[idx] - all_n_to_prune[idx-1]
    assert len(n_to_prune_sequence_head) == len(n_to_prune_sequence_intermediate)
    return n_to_prune_sequence_head, n_to_prune_sequence_intermediate


def calculate_head_and_intermediate_importance(
    model, 
    dataset,
    old_head_mask,
    old_intermediate_mask,
    trainer,
    normalize_scores_by_layer=True,
    disable_progress_bar=False,
    subset_size=1.0,

):
    training_flag = model.training
    model = model.module if hasattr(model, 'module') else model
    model.eval() 

    n_layers, n_heads, n_intermediate = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.intermediate_size
    head_importance = torch.zeros(n_layers, n_heads).to(old_head_mask)
    head_mask = torch.ones(n_layers, n_heads).to(old_head_mask)[:] = old_head_mask.clone()
    head_mask.requires_grad_(requires_grad=True)
    intermediate_importance = torch.zeros(n_layers, n_intermediate).to(old_intermediate_mask)
    intermediate_mask = torch.ones(n_layers, n_intermediate).to(old_intermediate_mask)[:] = old_intermediate_mask.clone()
    intermediate_mask.requires_grad_(requires_grad=True)

    batch_size = trainer.args.train_batch_size
    if subset_size <= 1:
        subset_size *= len(dataset)
    n_prune_steps = int(np.ceil(int(subset_size) / batch_size))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.data_collator,
        drop_last=trainer.args.dataloader_drop_last,
        num_workers=trainer.args.dataloader_num_workers,
        pin_memory=trainer.args.dataloader_pin_memory,
    )
    dataloader = islice(dataloader, n_prune_steps)
    prune_iterator = tqdm(
        dataloader,
        desc="Iteration",
        disable=disable_progress_bar,
        total=n_prune_steps
    )

    for inputs in prune_iterator:
        # key: add head mask and intermediate mask, so we can get the gradients from them
        inputs['head_mask'] = head_mask 
        inputs['intermediate_mask'] = intermediate_mask
        inputs = trainer._prepare_inputs(inputs)
        loss = trainer.compute_loss(model, inputs)
        loss.backward()
        head_importance += head_mask.grad.abs().detach()
        intermediate_importance += intermediate_mask.grad.abs().detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0)
        torch.nn.utils.clip_grad_norm_(head_mask, 0)
        torch.nn.utils.clip_grad_norm_(intermediate_mask, 0)
    
    if normalize_scores_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        norm_by_layer = torch.pow(torch.pow(intermediate_importance, exponent).sum(-1), 1/exponent)
        intermediate_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
    
    if training_flag:
        model.train()

    return head_importance, intermediate_importance

def what_to_prune_head(
    head_importance,
    n_to_prune,
    old_head_mask,
    at_least_x_heads_per_layer=1,
):
    head_importance = head_importance.clone()
    n_layers, n_heads = head_importance.size()

    already_prune = {}
    for layer in range(old_head_mask.size(0)):
        for head in range(old_head_mask.size(1)):
            if old_head_mask[layer][head].item() == 0:
                if layer not in already_prune:
                    already_prune[layer] = []
                already_prune[layer].append(head)

    # Sort heads by score
    heads_and_score = [
        ((layer, head), head_importance[layer][head].item())
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
    sorted_heads = [head_and_score[0]
                    for head_and_score in heads_and_score]
    # Ensure we don't delete all heads in a layer
    if at_least_x_heads_per_layer:
        # Remove the top scoring head in each layer
        to_protect = {l: 0 for l in range(n_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < at_least_x_heads_per_layer:
                    to_protect[layer] += 1
                    continue
                else:
                    to_protect.pop(layer)
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [
        (layer, head)
        for (layer, head) in sorted_heads
        if layer not in already_prune or head not in already_prune[layer]
    ]

    old_head_mask = old_head_mask.clone()
    new_head_mask = old_head_mask.clone()
    # Update heads to prune
    for layer, head in sorted_heads[:n_to_prune]:
        new_head_mask[layer][head] = 0
    return new_head_mask

def what_to_prune_mlp(
    intermediate_importance,
    n_to_prune,
    old_intermediate_mask
):
    intermediate_importance = intermediate_importance.clone()
    n_layers, n_intermediate = intermediate_importance.size()

    already_prune = defaultdict(list)
    for layer in range(n_layers):
        for intermediate_idx in range(n_intermediate):
            if old_intermediate_mask[layer][intermediate_idx].item() == 0:
                already_prune[layer].append(intermediate_idx)

    score = [
        ((layer, intermediate_idx), intermediate_importance[layer][intermediate_idx].item()) 
        for layer in range(n_layers) for intermediate_idx in range(n_intermediate)
    ]
    score.sort(key=lambda x:x[-1])
    filter_score = [
        ((layer, intermediate_idx), score)
        for ((layer, intermediate_idx), score) in score
        if layer not in already_prune or intermediate_idx not in already_prune[layer]
    ]

    old_intermediate_mask = old_intermediate_mask.clone()
    new_intermediate_mask = old_intermediate_mask.clone()
    for (layer, intermediate_idx), _ in filter_score[:n_to_prune]:
        new_intermediate_mask[layer][intermediate_idx] = 0
    return new_intermediate_mask
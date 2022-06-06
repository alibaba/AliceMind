# coding=utf-8
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2020 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import types
from typing import Iterable
import torch
from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
import math
from torch.distributions.bernoulli import Bernoulli
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from .compat import _report_compat_error
from .file_utils import logging

_report_compat_error()
sofa_backend = os.environ["SOFA_BACKEND"]

logger = logging.get_logger(__name__)


def calculate_fisher(trainer, reserve_p):
    '''
    Calculate Fisher Information for different parameters
    '''
    _report_compat_error()
    if sofa_backend == "huggingface":
        def report_trainer_param(trainer_, element):
            if not hasattr(trainer_, element):
                raise RuntimeError(f"No {element} attr found in trainer, please make sure"
                                   "trainer is transformers.Trainer or the version of huggingface is right.")

        report_trainer_param(trainer, "_prepare_inputs")
        report_trainer_param(trainer, "compute_loss")
        report_trainer_param(trainer, "model")
        report_trainer_param(trainer, "get_train_dataloader")
        prepare_inputs = trainer._prepare_inputs
        compute_loss = trainer.compute_loss
        model = trainer.model
        train_dataloader = trainer.get_train_dataloader()
        max_grad_norm = 1.0  # a default value
        if hasattr(trainer, "args") and hasattr(trainer.args, "max_grad_norm"):
            max_grad_norm = trainer.args.max_grad_norm if trainer.args.max_grad_norm is not None \
                                                          and trainer.args.max_grad_norm > 0 else max_grad_norm
    elif sofa_backend in ["easytexminer", "easynlp"]:
        if sofa_backend == "easytexminer":
            from easytexminer.utils import get_args
        else:
            from easynlp.utils import get_args

        def report_trainer_param(trainer_, element):
            if not hasattr(trainer_, element):
                raise RuntimeError(f"No {element} attr found in trainer, please make sure"
                                   f"trainer is {sofa_backend}.Trainer.")

        def prepare_inputs(batch):
            args = get_args()
            batch = {
                key: val.to(args.local_rank) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            return batch

        def compute_loss(model_, inputs_):
            label_ids = inputs_.pop("label_ids")
            forward_outputs = model_(inputs)
            return model_.compute_loss(forward_outputs, label_ids)

        report_trainer_param(trainer, "_model")
        report_trainer_param(trainer, "_train_loader")
        model = trainer._model
        train_dataloader = trainer._train_loader
        args = get_args()
        max_grad_norm = 1.0  # a default value
        if hasattr(args, "max_grad_norm"):
            max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None \
                                                  and args.max_grad_norm > 0 else max_grad_norm
    else:
        return

    gradient_mask = dict()
    model.train()
    for name, params in model.named_parameters():
        if 'layer' in name:
            gradient_mask[params] = params.new_zeros(params.size())

    N = len(train_dataloader)
    for inputs in tqdm(train_dataloader):
        if sofa_backend == "huggingface":
            if "idx" in inputs:
                inputs.pop("idx")
            inputs = prepare_inputs(inputs)
            loss = compute_loss(model, inputs)
        elif sofa_backend in ["easytexminer", "easynlp"]:
            inputs = prepare_inputs(inputs)
            outputs = compute_loss(model, inputs)
            loss = outputs["loss"]
        else:
            return

        loss.backward()
        for name, params in model.named_parameters():
            if 'layer' in name:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                gradient_mask[params] += (params.grad ** 2) / N
        model.zero_grad()

    logger.info('Calculate Fisher Information...')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1 - reserve_p) * 100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar
    print('Polar => {}'.format(polar))

    # TODO: pytorch: torch.kthvalue

    return gradient_mask


def step_adamw(self, closure: Callable = None):
    """
    Performs a single optimization step.
    Arguments:
        closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()
    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

            # =================== HACK BEGIN =====================
            if self.mode is not None:
                if self.mode == 'ChildTuning-D':
                    if p in self.gradient_mask:
                        grad *= self.gradient_mask[p]
                else:
                    # ChildTuning-F
                    grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                    grad *= grad_mask.sample() / self.reserve_p
            # =================== HACK END =======================

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
            if group["correct_bias"]:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group["weight_decay"] > 0.0:
                p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

    return loss


def step_adam(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError(
                    'Adam does not support sparse gradients, please consider SparseAdam instead'
                )

            # =================== HACK BEGIN =====================
            if self.mode is not None:
                if self.mode == 'ChildTuning-D':
                    if p in self.gradient_mask:
                        grad *= self.gradient_mask[p]
                else:
                    # ChildTuning-F
                    grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                    grad *= grad_mask.sample() / self.reserve_p
            # =================== HACK END =======================

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['next_m'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['next_v'] = torch.zeros_like(p.data)

            next_m, next_v = state['next_m'], state['next_v']
            beta1, beta2 = group['b1'], group['b2']

            # Add grad clipping
            if group['max_grad_norm'] > 0:
                clip_grad_norm_(p, group['max_grad_norm'])

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            next_m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            next_v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            update = next_m / (next_v.sqrt() + group['e'])

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if group['weight_decay'] > 0.0:
                update += group['weight_decay'] * p.data

            lr_scheduled = group['lr']
            lr_scheduled *= group['schedule'].get_lr(state['step'])

            update_with_lr = lr_scheduled * update
            p.data.add_(-update_with_lr)

            state['step'] += 1

            # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
            # No bias correction
            # bias_correction1 = 1 - beta1 ** state['step']
            # bias_correction2 = 1 - beta2 ** state['step']

    return loss


class ChildTuningAdamW(Optimizer):
    """
    A ChildTuning AdamW optimizer, user can directly use it.
    """
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            reserve_p=0.2,
            mode="ChildTuning-F"
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask


ChildTuningAdamW.step = step_adamw


def apply_child_tuning(mode="ChildTuning-F", reserve_p=0.2):
    """
    Apply child tuning to all trainer classes.
    :param mode: The child_tuning type. Support: "ChildTuning-F" or "ChildTuning-D"
    :param reserve_p: The reserved gradiant ratio.
    :return: None
    """
    if mode == "ChildTuning-D":
        raise RuntimeError("Mode ChildTuning-D is task related, please use ChildTuning-F mode.")
    if sofa_backend == "huggingface":
        from transformers import AdamW
        AdamW.mode = mode
        AdamW.reserve_p = reserve_p
        AdamW.step = step_adamw
    elif sofa_backend in ["easytexminer", "easynlp"]:
        if sofa_backend == "easytexminer":
            from easytexminer.core.optimizers import BertAdam
        else:
            from easynlp.core.optimizers import BertAdam
        BertAdam.mode = mode
        BertAdam.reserve_p = reserve_p
        BertAdam.step = step_adam
    elif sofa_backend == "sofa":
        from .backend import AdamW
        AdamW.mode = mode
        AdamW.reserve_p = reserve_p
        AdamW.step = step_adamw
    else:
        _report_compat_error()


def apply_child_tuning_to_trainer(trainer,
                                  mode="ChildTuning-F",
                                  reserve_p=0.2):
    """
    Apply child tuning to a trainer instance.
    :param trainer: The trainer instance.
    :param mode: The child_tuning type. Support: "ChildTuning-F" or "ChildTuning-D"
    :param reserve_p: The reserved gradiant ratio.
    :return: None
    """
    gradient_mask = None
    if mode == "ChildTuning-D":
        gradient_mask = calculate_fisher(trainer, reserve_p)
    if sofa_backend in ["huggingface", "sofa"]:
        if sofa_backend == "huggingface":
            from transformers import AdamW, TrainerCallback
        else:
            from .backend import AdamW, TrainerCallback

        class OnTrainBeginCallback(TrainerCallback):
            def on_train_begin(self, *args, **kwargs):
                optimizer = kwargs["optimizer"]
                if type(optimizer) != AdamW:
                    raise RuntimeError(f"Only AdamW is supported, not {type(optimizer)}.")
                optimizer.mode = mode
                optimizer.reserve_p = reserve_p
                optimizer.gradient_mask = gradient_mask
                optimizer.step = types.MethodType(step_adamw, optimizer)
        trainer.callback_handler.callbacks.append(OnTrainBeginCallback())
    elif sofa_backend in ["easytexminer", "easynlp"]:
        if sofa_backend == "easytexminer":
            from easytexminer.core.optimizers import BertAdam
        else:
            from easynlp.core.optimizers import BertAdam
        if not hasattr(trainer, "_optimizer"):
            raise RuntimeError("No optimizer found in trainer, please check the input param "
                               "or the version of easytexminer.")
        optimizer = getattr(trainer, "_optimizer")
        if type(optimizer) != BertAdam:
            raise RuntimeError(f"Only BertAdam is supported, not {type(optimizer)}.")
        optimizer.mode = mode
        optimizer.reserve_p = reserve_p
        optimizer.gradient_mask = gradient_mask
        optimizer.step = types.MethodType(step_adam, optimizer)
    else:
        _report_compat_error()

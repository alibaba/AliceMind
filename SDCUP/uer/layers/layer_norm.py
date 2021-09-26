# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class LayerNormOri(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNormOri, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.ori_norm = torch.nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        # print('ori norm!')
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.ori_norm(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        # print('uer norm!')


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta


class LayerNormSQL(nn.Module):
    def __init__(self, hS, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNormSQL, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hS))
        self.beta = nn.Parameter(torch.zeros(hS))
        self.variance_epsilon = variance_epsilon
        # print('norm from sqlova!!')


    def forward(self, x):
        # print('sqlova norm!!!')
        # x = input to the neuron.
        # normalize each vector (each token).
        # regularize x.
        # If x follows Gaussian distribution, it becomes standard Normal distribution (i.e., mu=0, std=1).
        u = x.mean(-1, keepdim=True)  # keepdim = keeprank of tensor.
        s = (x - u).pow(2).mean(-1, keepdim=True) # variance
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # standard
        # x = x.to(device)
        # Gamma & Beta is trainable parameters.
        return self.gamma * x + self.beta



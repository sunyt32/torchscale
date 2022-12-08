# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
from scipy.optimize import minimize

import torch
import torch.nn as nn

def fixed_pos_embedding(x, offset=0):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(offset, seq_len + offset, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def get_scale(dim, scale_base = 256):
    x = np.arange(0, 4096, 1)
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    angle = x[:, None] * inv_freq
    scale_init = -np.log(1 / ((np.arange(0, dim, 2) + 1 * dim) / (2 * dim)) - 1)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def eval_fun(scale):
        scale = sigmoid(scale)
        posi_scale = (scale ** (x[:, None] / scale_base))
        upper_bound = (np.cos(angle) * posi_scale).mean(1) # \sum_i cos n\theta_i p^{n}
        delta = (upper_bound[:-1] - upper_bound[1:]) / upper_bound[:-1] # \sum_n (f(n) - f(n+1)) / f(n)
        return delta

    res = minimize(lambda scale: -eval_fun(scale).sum(), scale_init)
    return sigmoid(res.x)

class SoPE(nn.Module):
    def __init__(
        self, head_dim, scale_base = 256
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", torch.from_numpy(get_scale(self.head_dim, scale_base)).float()
        )

    def forward(self, len, offset=0):
        scale = self.scale ** torch.arange(offset, len + offset, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale, offset=offset)
        return (sin, cos, scale)

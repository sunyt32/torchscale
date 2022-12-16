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


class SoPE(nn.Module):
    def __init__(
        self, head_dim, scale_base = 512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.35 * head_dim) / (1.35 * head_dim)
        )

    def forward(self, len, offset=0):
        scale = self.scale.float()
        scale = scale ** torch.arange(offset, len + offset, 1).to(scale).div(self.scale_base)[:, None]
        scale = (scale * (3e-5 / scale.min())).to(self.scale)
        sin, cos = fixed_pos_embedding(scale, offset=offset)
        return (sin, cos, scale)

"""
utility functions for CISO model
"""

import math

import torch
from torch import nn


def weights_init(module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def custom_replace_n(data: torch.tensor, n: int):
    """
    replacing unique values with their index
    """
    original_values = torch.linspace(1 / n, 1, n)  # Values from 1/n to 1

    original_values = torch.cat((torch.tensor([-1, 0]), original_values), dim=0)
    new_values = torch.arange(0, n + 2, 1)
    res = data.clone()
    for original_value, new_value in zip(original_values, new_values):
        mask = torch.isclose(data, original_value, atol=1e-3)
        res[mask] = new_value

    return res

"""
NN models
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.utils import custom_replace_n


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class SimpleMLPMasked_v1(nn.Module):
    """
    Simple MLP Masked where env features and mask use seperate encoders
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        quantized_mask_bins=1,
    ):
        super(SimpleMLPMasked_v1, self).__init__()

        self.num_unique_mask_values = quantized_mask_bins

        self.mask_encoder = SimpleMLPBackbone(input_dim=((self.num_unique_mask_values + 2) * num_classes), hidden_dim=hidden_dim, num_layers=2)

        self.env_encoder = SimpleMLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2)

        self.out_layer = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask_q):
        mask_q[mask_q == -2] = -1
        mask_q = torch.where(
            mask_q > 0,
            torch.ceil(mask_q * self.num_unique_mask_values)
            / self.num_unique_mask_values,
            mask_q,
        )
        mask_q = custom_replace_n(mask_q, self.num_unique_mask_values).long()
        one_hot_mask = F.one_hot(
            mask_q, num_classes=self.num_unique_mask_values + 2
        ).float()  # One-hot encoding
        one_hot_mask_flattened = one_hot_mask.view(
            mask_q.size(0), -1
        )  # Flatten to (batch_size, num_classes * 3)

        x_env = self.env_encoder(x)
        x_mask = self.mask_encoder(one_hot_mask_flattened)
        x_combined = torch.cat((x_env, x_mask), dim=1)
        x = self.out_layer(x_combined)
        return x


class SimpleMLPMasked_v0(nn.Module):
    """
    Simple MLP Masked where env features and mask share the same encoder
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        backbone=None,
        attention_layers=2,
        heads=2,
        num_unique_mask_values=3,
    ):
        super(SimpleMLPMasked_v0, self).__init__()

        self.num_unique_mask_values = num_unique_mask_values

        self.layer_1 = nn.Linear(input_dim + (self.num_unique_mask_values * num_classes), hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        mask = mask.long()
        one_hot_mask = F.one_hot(mask, num_classes=self.num_unique_mask_values).float()  # One-hot encoding
        one_hot_mask_flattened = one_hot_mask.view(mask.size(0), -1)  # Flatten to (batch_size, num_classes * 3)

        x_combined = torch.cat((x, one_hot_mask_flattened), dim=1)
        x = F.relu(self.layer_1(x_combined))
        x = F.relu(self.layer_2(x))
        x = self.out_layer(x)
        return x


class SimpleMLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(SimpleMLPBackbone, self).__init__()
        self.num_layers = num_layers
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        for i in range(1, num_layers):
            setattr(self, f"layer_{i+1}", nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        for i in range(1, self.num_layers - 1):
            x = F.relu(getattr(self, f"layer_{i+1}")(x))
        x = getattr(self, f"layer_{self.num_layers}")(x)
        return x


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, activation="relu"
        )

    def forward(self, k, mask=None):
        k = k.transpose(0, 1)
        x = self.transformer_layer(k, src_mask=mask)
        x = x.transpose(0, 1)
        return x

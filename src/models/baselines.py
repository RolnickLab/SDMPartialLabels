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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class SimpleMLPMasked(nn.Module):
    def __init__(
        self,
        input_channels,
        d_hidden,
        num_classes,
        quantized_mask_bins=1,
        backbone=None,
    ):
        super(SimpleMLPMasked, self).__init__()

        self.num_unique_mask_values = quantized_mask_bins

        self.layer_1 = nn.Linear(input_channels + ((self.num_unique_mask_values + 2) * num_classes), d_hidden)
        self.layer_2 = nn.Linear(d_hidden, d_hidden)
        self.out_layer = nn.Linear(d_hidden, num_classes)

    def forward(self, x, mask_q):
        mask_q[mask_q == -2] = -1
        mask_q = torch.where(mask_q > 0, torch.ceil(mask_q * self.num_unique_mask_values) / self.num_unique_mask_values, mask_q)
        mask_q = custom_replace_n(mask_q, self.num_unique_mask_values).long()
        one_hot_mask = F.one_hot(mask_q, num_classes=self.num_unique_mask_values + 2).float()  # One-hot encoding
        one_hot_mask_flattened = one_hot_mask.view(mask_q.size(0), -1)  # Flatten to (batch_size, num_classes * 3)

        x_combined = torch.cat((x, one_hot_mask_flattened), dim=1)
        x = F.relu(self.layer_1(x_combined))
        x = F.relu(self.layer_2(x))
        x = self.out_layer(x)
        return x


class SimpleMLPBackbone(nn.Module):
    def __init__(self, input_channels, pretrained=False, hidden_dim=64):
        super(SimpleMLPBackbone, self).__init__()
        self.layer_1 = nn.Linear(input_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class PredHead(nn.Module):

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return x


class MlpEncoder(nn.Module):
    class Block(nn.Module):

        def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self, d_in: int, d_out: int, n_layers: int, dropout: float) -> None:
        super(MlpEncoder, self).__init__()
        self.blocks = nn.Sequential(
            *[
                self.Block(d_in if i == 0 else d_out, d_out, dropout)
                for i in range(n_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp_encoder = MlpEncoder(
            d_in=input_dim, d_out=hidden_dim, n_layers=n_layers, dropout=dropout
        )
        self.head = PredHead(d_in=hidden_dim, d_out=output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.mlp_encoder(x))


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

"""
NN models
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from src.models.utils import init_first_layer_weights


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

class SimpleMLPBackbone(nn.Module):
    def __init__(self, input_channels, pretrained=False, hidden_dim=64, num_layers=2):
        super(SimpleMLPBackbone, self).__init__()
        self.num_layers = num_layers
        self.layer_1 = nn.Linear(input_channels, hidden_dim)
        for i in range(1, num_layers):
            setattr(self, f'layer_{i+1}',  nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        for i in range(1, self.num_layers-1):
            x = F.relu(getattr(self, f'layer_{i+1}')(x))
        x = getattr(self, f'layer_{self.num_layers}')(x)
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


class Resnet18(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(Resnet18, self).__init__()
        self.x_input_channels = input_channels
        self.pretrained_backbone = pretrained

        self.freeze_base = False
        self.unfreeze_base_l4 = False

        self.base_network = models.resnet18(pretrained=self.pretrained_backbone)
        original_in_channels = self.base_network.conv1.in_channels

        # if input is not RGB
        if self.x_input_channels != original_in_channels:
            original_weights = self.base_network.conv1.weight.data.clone()
            self.base_network.conv1 = nn.Conv2d(
                self.x_input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            if self.pretrained_backbone:
                self.base_network.conv1.weight.data[:, :original_in_channels, :, :] = (
                    original_weights
                )
                self.base_network.conv1.weight.data = init_first_layer_weights(
                    self.x_input_channels, original_weights
                )

        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False

    def forward(self, images):
        x = self.base_network.conv1(images)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)
        # x = self.base_network.avgpool(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(Resnet50, self).__init__()
        self.x_input_channels = input_channels
        self.pretrained_backbone = pretrained

        self.freeze_base = False
        self.unfreeze_base_l4 = False

        self.base_network = models.resnet50(pretrained=self.pretrained_backbone)
        original_in_channels = self.base_network.conv1.in_channels

        # if input is not RGB
        if self.x_input_channels != original_in_channels:
            original_weights = self.base_network.conv1.weight.data.clone()
            self.base_network.conv1 = nn.Conv2d(
                self.x_input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            if self.pretrained_backbone:
                self.base_network.conv1.weight.data[:, :original_in_channels, :, :] = (
                    original_weights
                )
                self.base_network.conv1.weight.data = init_first_layer_weights(
                    self.x_input_channels, original_weights
                )

        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False
        elif self.unfreeze_base_l4:
            for p in self.base_network.layer4.parameters():
                p.requires_grad = True

        # TODO: use avgpool layer: original is AdaptiveAvgPool2d(output_size=(1, 1))
        self.base_network.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        x = self.base_network.conv1(images)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)
        return x


class Resnet101(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(Resnet101, self).__init__()
        self.x_input_channels = input_channels
        self.pretrained_backbone = pretrained

        self.freeze_base = False
        self.unfreeze_base_l4 = False

        self.base_network = models.resnet101(pretrained=self.pretrained_backbone)
        original_in_channels = self.base_network.conv1.in_channels

        # if input is not RGB
        if self.x_input_channels != original_in_channels:
            original_weights = self.base_network.conv1.weight.data.clone()
            self.base_network.conv1 = nn.Conv2d(
                self.x_input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            if self.pretrained_backbone:
                self.base_network.conv1.weight.data[:, :original_in_channels, :, :] = (
                    original_weights
                )
                self.base_network.conv1.weight.data = init_first_layer_weights(
                    self.x_input_channels, original_weights
                )

        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False
        elif self.unfreeze_base_l4:
            for p in self.base_network.layer4.parameters():
                p.requires_grad = True

        self.base_network.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        x = self.base_network.conv1(images)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)
        # x = self.base_network.avgpool(x)
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

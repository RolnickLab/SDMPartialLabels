import torch
import torch.nn.functional as F
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class MLPBackbone(nn.Module):
    def __init__(self, input_channels, pretrained=False, hidden_dim=64):
        super(MLPBackbone, self).__init__()
        self.layer_1 = nn.Linear(input_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_channels,
        d_hidden,
        num_classes,
        backbone=None,
        attention_layers=2,
        heads=2,
    ):
        super(MaskedMLP, self).__init__()
        self.layer_1 = nn.Linear(input_channels + num_classes, d_hidden)
        self.layer_2 = nn.Linear(d_hidden, d_hidden)
        self.layer_3 = nn.Linear(d_hidden, num_classes)

    def forward(self, x, mask):
        x_combined = torch.cat((x, mask), dim=1)

        x = F.relu(self.layer_1(x_combined))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

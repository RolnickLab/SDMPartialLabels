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
        num_unique_mask_values=3,
    ):
        super(MaskedMLP, self).__init__()
        self.num_unique_mask_values = num_unique_mask_values

        self.env_encoder = nn.Sequential(
            nn.Linear(input_channels, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(self.num_unique_mask_values * num_classes, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.out_layer = nn.Linear(d_hidden*2, num_classes)

    def forward(self, x, mask):
        mask = mask.long()
        one_hot_mask = F.one_hot(mask, num_classes=self.num_unique_mask_values).float()  # One-hot encoding
        one_hot_mask_flattened = one_hot_mask.view(mask.size(0), -1)  # Flatten to (batch_size, num_classes * 3)
        mask_encoded = self.mask_encoder(one_hot_mask_flattened)

        x_encoded = self.env_encoder(x)
        x_combined = torch.cat((x_encoded, mask_encoded), dim=1)
        x = self.out_layer(x_combined)
        return x


class SimpleMLPMasked(nn.Module):
    def __init__(
        self,
        input_channels,
        d_hidden,
        num_classes,
        backbone=None,
        attention_layers=2,
        heads=2,
        num_unique_mask_values=3,
    ):
        super(SimpleMLPMasked, self).__init__()

        self.num_unique_mask_values = num_unique_mask_values

        self.layer_1 = nn.Linear(input_channels + (self.num_unique_mask_values * num_classes), d_hidden)
        self.layer_2 = nn.Linear(d_hidden, d_hidden)
        self.out_layer = nn.Linear(d_hidden, num_classes)

    def forward(self, x, mask):
        mask = mask.long()
        one_hot_mask = F.one_hot(mask, num_classes=self.num_unique_mask_values).float()  # One-hot encoding
        one_hot_mask_flattened = one_hot_mask.view(mask.size(0), -1)  # Flatten to (batch_size, num_classes * 3)

        x_combined = torch.cat((x, one_hot_mask_flattened), dim=1)
        x = F.relu(self.layer_1(x_combined))
        x = F.relu(self.layer_2(x))
        x = self.out_layer(x)
        return x

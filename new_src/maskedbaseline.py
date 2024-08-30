import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_channels,
        d_hidden,
        num_classes,
        latent_dim=32,
        backbone=None,
        attention_layers=2,
        heads=2,
    ):
        super(MaskedMLP, self).__init__()

        self.input_size = input_channels
        self.num_classes = num_classes

        # Encoder network
        self.encoder_env = nn.Sequential(
            nn.Linear(self.input_size, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, latent_dim),
            nn.ReLU(),
        )
        self.encoder_labels = nn.Sequential(
            nn.Linear(num_classes, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, latent_dim),
            nn.ReLU(),
        )

        # Class prediction layer
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, self.num_classes),
        )

    def forward(self, x, mask):
        # Combine the input features with the mask
        encoded_x = self.encoder_env(x)
        encoded_labels = self.encoder_labels(mask.float())

        feat_combined = torch.cat((encoded_x, encoded_labels), dim=1)

        # Predict the class probabilities
        predicted_classes = self.classifier(feat_combined)

        return predicted_classes


class MaskedMLP_2(nn.Module):
    def __init__(
        self,
        input_channels,
        d_hidden,
        num_classes,
        backbone=None,
        attention_layers=2,
        heads=2,
    ):
        super(MaskedMLP_2, self).__init__()
        self.layer_1 = nn.Linear(input_channels + num_classes, d_hidden)
        self.layer_2 = nn.Linear(d_hidden, d_hidden)
        self.layer_3 = nn.Linear(d_hidden, num_classes)

    def forward(self, x, mask):
        x_combined = torch.cat((x, mask), dim=1)

        x = F.relu(self.layer_1(x_combined))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

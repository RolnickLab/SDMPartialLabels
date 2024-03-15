import torch
import torchvision
import torch.nn as nn
from src.trainer.utils import init_first_layer_weights


class MultiInputResnet18(nn.Module):
    def __init__(self, img_input_channels, env_input_channels, target_size,pretrained=True):
        super(MultiInputResnet18, self).__init__()

        self.pretrained_backbone = pretrained
        self.img_input_channels = img_input_channels
        self.env_input_channels = env_input_channels
        self.target_size = target_size

        self.base_network_1 = torchvision.models.resnet18(pretrained=self.pretrained_backbone)
        self.base_network_2 = torchvision.models.resnet18(pretrained=self.pretrained_backbone)
        original_in_channels = self.base_network_1.conv1.in_channels
        original_weights = self.base_network_1.conv1.weight.data.clone()

        # if input is not RGB
        self.base_network_1.conv1 = nn.Conv2d(img_input_channels, 64, kernel_size=(7, 7),
                                            stride=(2, 2),
                                            padding=(3, 3),
                                            bias=False,
                                            )
        if self.pretrained_backbone:
            self.base_network_1.conv1.weight.data[:, :original_in_channels, :, :] = original_weights
            self.base_network_1.conv1.weight.data = init_first_layer_weights(self.img_input_channels,
                                                                           original_weights)


        self.base_network_2.conv1 = nn.Conv2d(env_input_channels, 64, kernel_size=(7, 7),
                                            stride=(2, 2),
                                            padding=(3, 3),
                                            bias=False,
                                            )

        if self.pretrained_backbone:
            self.base_network_2.conv1.weight.data[:, :original_in_channels, :, :] = original_weights
            self.base_network_2.conv1.weight.data = init_first_layer_weights(self.env_input_channels,
                                                                           original_weights)


        self.linear = nn.Linear(1024, self.target_size)

    def forward(self, inp):
        img_data = inp[:, 0:self.img_input_channels, :, :]
        env_data = inp[:, self.img_input_channels:, :, :]

        x = self.base_network_1.conv1(img_data)
        x = self.base_network_1.bn1(x)
        x = self.base_network_1.relu(x)
        x = self.base_network_1.maxpool(x)
        x = self.base_network_1.layer1(x)
        x = self.base_network_1.layer2(x)
        x = self.base_network_1.layer3(x)
        x = self.base_network_1.layer4(x)
        x_1 = self.base_network_1.avgpool(x)

        x = self.base_network_2.conv1(env_data)
        x = self.base_network_2.bn1(x)
        x = self.base_network_2.relu(x)
        x = self.base_network_2.maxpool(x)
        x = self.base_network_2.layer1(x)
        x = self.base_network_2.layer2(x)
        x = self.base_network_2.layer3(x)
        x = self.base_network_2.layer4(x)
        x_2 = self.base_network_2.avgpool(x)

        out = torch.cat([x_1, x_2], dim=1)
        out = torch.reshape(out, [out.size(0), out.size(1)])

        out = self.linear(out)

        return out
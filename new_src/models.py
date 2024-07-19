import logging

import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        logging.info("x size: ", x.size())
        x = F.relu(self.layer_1(x))
        logging.info("x size: ", x.size())
        x = F.relu(self.layer_2(x))
        logging.info("x size: ", x.size())
        x = self.layer_3(x)
        logging.info("x size: ", x.size())
        return x

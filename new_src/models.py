import torch.nn.functional as F
from torch import Tensor, nn


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

#adapted from MaskedSDM paper
import torch
import math

from typing import Union

from torch import nn, Tensor

class SpeciesTokenizer(nn.Module):
    
    def __init__(self, n_species: int, d_token: int, tokenization: str = "periodic") -> None:
        super().__init__()
        self.n_species = n_species
        self.d_token = d_token
        self.tokenization = tokenization
        
        if self.tokenization == "periodic":
            self.periodic_embeddings = PeriodicEmbeddings(n_species=n_species, d_embedding=d_token)
        elif self.tokenization == "linear":
            self.weight = nn.Parameter(Tensor(n_species, d_token))
            self.bias = nn.Parameter(Tensor(n_species, d_token))
            for parameter in [self.weight, self.bias]:
                nn.init.uniform_(parameter, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        elif self.tokenization == "categorical":
            self.pos_embeddings = nn.Parameter(n_species, d_token)
            for parameter in [self.pos_embeddings]:
                nn.init.uniform_(parameter, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
            self.neg_embeddings = nn.Parameter(n_species, d_token)
            for parameter in [self.neg_embeddings]:
                nn.init.uniform_(parameter, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
            self.unknown_embeddings = nn.Parameter(n_species, d_token)
            for parameter in [self.unknown_embeddings]:
                nn.init.uniform_(parameter, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        else:
            raise ValueError(f"Invalid tokenization method: {self.tokenization}")
            
        
    def forward(self, x: Tensor) -> Tensor:
        if self.tokenization == "periodic":
            return self.periodic_embeddings(x)
        elif self.tokenization == "linear":
            return self.weight[None] * x[..., None] + self.bias[None]
        elif self.tokenization == "categorical":
            return torch.where(x == 0, self.neg_embeddings[None], torch.where(x == 1, self.pos_embeddings[None], self.unknown_embeddings[None]))
        else:
            raise ValueError(f"Invalid tokenization method: {self.tokenization}")
    
    
class _Periodic(nn.Module):

    def __init__(self, n_species: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = nn.Parameter(torch.empty(n_species, k))
        self.reset_parameters()

    def reset_parameters(self):
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x = x + self.bias
        return x

    
class PeriodicEmbeddings(nn.Module):

    def __init__(
        self,
        n_species: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01
    ) -> None:
        """
        Args:
            n_species: the number of species.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each species.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**,
                see the documentation for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_species, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        self.linear = _NLinear(n_species, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = self.periodic(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
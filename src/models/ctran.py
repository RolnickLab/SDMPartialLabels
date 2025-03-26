"""
Regression-Transformer model
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""

import math

import numpy as np
import torch
import torch.nn as nn

from src.models.baselines import SelfAttnLayer
from src.models.state_embeddings import SpeciesTokenizer
from src.models.utils import custom_replace_n, weights_init


class CTranModel(nn.Module):
    def __init__(
        self,
        num_classes,
            backbone="SimpleMLPBackbone",
            quantized_mask_bins=4,
        input_dim=27,
            hidden_dim=256,
        n_attention_layers=3,
        n_heads=4,
        dropout=0.2,
        n_backbone_layers=2,
        tokenize_state=None,
        use_unknown_token=False,
    ):
        """
        pos_emb is false by default
        num_classes: total number of species
        species_list: list of species
        backbone: backbone to process input (MLP)
        pretrained_backbone: to load ImageNet pretrained weights for backbone
        quantized_mask_bins (should be >= 1): how many bins to use for the positive encounter rate > 0
        input_dim: number of input channels for satellite data
        hidden_dim: embedding dimension / hidden layer dimention
        n_attention_layers: number of attention layes
        n_heads: number of attention heads
        dropout: dropout ratio
        use_unknown_token: add special parameter to encode unknown state when state is linearly tokenized
        """
        super(CTranModel, self).__init__()
        self.hidden_dim = hidden_dim  # this should match the backbone output feature size (512 for Resnet18, 2048 for Resnet50)
        self.quantized_mask_bins = quantized_mask_bins
        self.n_embedding_state = self.quantized_mask_bins + 2
        self.use_unknown_token = use_unknown_token
        self.backbone = globals()[backbone](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=n_backbone_layers,
        )

        # Env embed layer

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_classes)).view(1, -1).long()
        self.label_embeddings = torch.nn.Embedding(
            num_classes, self.hidden_dim, padding_idx=None
        )  # LxD

        # State Embeddings
        self.tokenize_state = tokenize_state
        if tokenize_state is not None:
            self.state_embeddings = SpeciesTokenizer(
                num_classes, hidden_dim, tokenization=self.tokenize_state
            )
            # tokens to symbolize unknown (instead of passing -1 to the species tokenizer, there is a special species specific mask token for unknown)
            self.mask_tokens = nn.Parameter(torch.Tensor(num_classes, hidden_dim))
            for parameter in [self.mask_tokens]:
                torch.nn.init.uniform_(
                    parameter, -1 / math.sqrt(hidden_dim), 1 / math.sqrt(hidden_dim)
                )
        else:
            self.state_embeddings = torch.nn.Embedding(
                self.n_embedding_state, self.hidden_dim, padding_idx=0
            )  # Dx2 (known, unknown)

        # Transformer
        self.self_attn_layers = nn.ModuleList(
            [
                SelfAttnLayer(self.hidden_dim, n_heads, dropout)
                for _ in range(n_attention_layers)
            ]
        )
        self.dense_layer = torch.nn.Linear(num_classes, self.hidden_dim)
        # Classifier
        # Output is of size num_classes because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(self.hidden_dim, num_classes)

        # Other
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_embeddings.apply(weights_init)
        self.state_embeddings.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, x_input, mask_q):
        x_input = x_input.type(torch.float32)
        x_features = self.backbone(x_input)  # image: HxWxD , out: [128, 4, 512]
        x_features = x_features.unsqueeze(1)
        const_label_input = self.label_input.repeat(x_input.size(0), 1).to(
            x_input.device
        )  # LxD (128, 670)
        init_label_embeddings = self.label_embeddings(
            const_label_input
        )  # LxD # (128, 670, 512)
        if self.quantized_mask_bins >= 1:
            mask_q[mask_q == -2] = -1
            mask_q = torch.where(
                mask_q > 0,
                torch.ceil(mask_q * self.quantized_mask_bins)
                / self.quantized_mask_bins,
                mask_q,
            )
            label_feat_vec = custom_replace_n(mask_q, self.quantized_mask_bins).long()
            state_embeddings = self.state_embeddings(label_feat_vec)  # (128, 670, 512)

        elif self.quantized_mask_bins == 0:
            mask_q[mask_q == -2] = -1
            # if self.tokenize_state, masks should have been constructed with quantized_bins=0
            # Get state embeddings
            state_embeddings = self.state_embeddings(mask_q)  # (128, 670, 512)
            if self.use_unknown_token:
                batch_size = state_embeddings.shape[0]
                unknown_tokens = self.mask_tokens.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                expanded_mask = (
                    (mask_q < 0).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                )  # Shape: (batch_size, num_classes, hidden_dim)
                state_embeddings = torch.where(
                    expanded_mask, unknown_tokens, state_embeddings
                )

        # Add state embeddings to label embeddings
        init_label_embeddings += state_embeddings
        # concatenate x_input features to label embeddings
        embeddings = torch.cat(
            (x_features, init_label_embeddings), 1
        )  # (128, 674, 512)
        # Feed all (image and label) embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        for layer in self.self_attn_layers:
            embeddings = layer(embeddings, mask=None)

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1) :, :]
        output = self.output_linear(label_embeddings)
        diag_mask = (
            torch.eye(output.size(1), device=output.device)
            .unsqueeze(0)
            .repeat(output.size(0), 1, 1)
        )
        output = (output * diag_mask).sum(-1)

        return output

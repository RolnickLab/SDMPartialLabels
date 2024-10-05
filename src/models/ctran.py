"""
Regression-Transformer model
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""

import numpy as np
import torch

from src.models.baselines import *
from src.models.utils import custom_replace, custom_replace_n, weights_init


class CTranModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="MlpEncoder",
        quantized_mask_bins=1,
        input_channels=1,
        n_layers=4,
        d_hidden=512,
        attention_layers=3,
        heads=4,
        dropout=0.2,
        use_pos_encoding=False,
    ):
        """
        pos_emb is false by default
        num_classes: total number of species
        species_list: list of species
        backbone: backbone to process input (MLP)
        pretrained_backbone: to load ImageNet pretrained weights for backbone
        quantized_mask_bins (should be >= 1): how many bins to use for the positive encounter rate > 0
        input_channels: number of input channels for satellite data
        d_hidden: embedding dimension / hidden layer dimention
        attention_layers: number of attention layes
        heads: number of attention heads
        dropout: dropout ratio
        use_pos_encoding: flag to use positional encoding or not
        """
        super(CTranModel, self).__init__()
        self.d_hidden = d_hidden  # this should match the backbone output feature size (512 for Resnet18, 2048 for Resnet50)
        self.use_pos_encoding = use_pos_encoding

        self.quantized_mask_bins = quantized_mask_bins
        self.n_embedding_state = self.quantized_mask_bins + 2

        self.backbone = globals()[backbone](
            d_in=input_channels, d_out=d_hidden, dropout=dropout, n_layers=n_layers
        )

        # Env embed layer

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_classes)).view(1, -1).long()
        self.label_embeddings = torch.nn.Embedding(
            num_classes, self.d_hidden, padding_idx=None
        )  # LxD

        # State Embeddings
        self.state_embeddings = torch.nn.Embedding(
            self.n_embedding_state, self.d_hidden, padding_idx=0
        )  # Dx2 (known, unknown)

        # Transformer
        self.self_attn_layers = nn.ModuleList(
            [
                SelfAttnLayer(self.d_hidden, heads, dropout)
                for _ in range(attention_layers)
            ]
        )
        self.dense_layer = torch.nn.Linear(num_classes, self.d_hidden)
        # Classifier
        # Output is of size num_classes because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(self.d_hidden, num_classes)

        # Other
        self.LayerNorm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_embeddings.apply(weights_init)
        self.state_embeddings.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, mask, mask_q=None):
        images = images.type(torch.float32)
        z_features = self.backbone(
            images.unsqueeze(-1)
        )  # image: HxWxD , out: [128, 4, 512]

        const_label_input = self.label_input.repeat(images.size(0), 1).to(
            images.device
        )  # LxD (128, 670)
        init_label_embeddings = self.label_embeddings(
            const_label_input
        )  # LxD # (128, 670, 512)

        mask[mask == -2] = -1
        if self.quantized_mask_bins > 1:
            mask_q[mask_q == -2] = -1
            label_feat_vec = custom_replace_n(mask_q).long()
        else:
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

        # Get state embeddings
        state_embeddings = self.state_embeddings(label_feat_vec)  # (128, 670, 512)
        # Add state embeddings to label embeddings
        init_label_embeddings += state_embeddings
        # concatenate images features to label embeddings
        embeddings = torch.cat(
            (z_features, init_label_embeddings), 1
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

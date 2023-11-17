"""
Regression-Transformer model
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""
import torch
import torch.nn as nn
import numpy as np
from Rtran.utils import *
from Rtran.models import *


class RTranModel(nn.Module):
    def __init__(self, num_classes, backbone='Resnet18', pretrained_backbone=True, quantized_mask_bins=1, input_channels=3, d_hidden=512, attention_layers=3, heads=4, dropout=0.2, use_pos_encoding=False, scale_embeddings_by_labels=False):
        """
        pos_emb is false by default
        """
        super(RTranModel, self).__init__()
        self.d_hidden = d_hidden  # this should match the backbone output feature size (512 for Resnet18, 2048 for Resnet50)
        self.scale_embeddings_by_labels = scale_embeddings_by_labels
        self.use_pos_encoding = use_pos_encoding
        self.use_lmt = True
        self.use_text_species = True

        self.quantized_mask_bins = quantized_mask_bins
        self.n_embedding_state = self.quantized_mask_bins + 2

        # ResNet101 backbone
        self.backbone = globals()[backbone](input_channels=input_channels, pretrained=pretrained_backbone)
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_classes)).view(1, -1).long()

        # word embeddings
        if self.use_text_species:
            padded_species, word_to_idx, vocab_size = tokenize_species() # 866, (670, 2)
            self.embedded_species = torch.tensor(padded_species, dtype=torch.long) # (670, 2)
            # TODO: modify the labels to actually take text labels rather than label numbers
            self.label_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.d_hidden, padding_idx=None)  # LxD
            # self.label_embeddings.weight.data.copy_(torch.from_numpy(load_word2vec_pretrained_weights(word_to_idx=word_to_idx, vocab_size=vocab_size, embedding_dim=self.d_hidden)))
        else:
            self.label_embeddings = torch.nn.Embedding(num_classes, self.d_hidden, padding_idx=None)  # LxD
        # print("layer: ", self.label_embeddings)

        # State Embeddings
        self.state_embeddings = torch.nn.Embedding(self.n_embedding_state, self.d_hidden, padding_idx=0) # Dx2 (known, unknown)

        # embedding for the regression labels
        # self.regression_embedding = torch.nn.Linear(num_classes, num_classes)
        # TODO: Position Embeddings (for image features)
        if self.use_pos_encoding:
            # self.position_encoding = positional_encoding_2d(2, 2, self.d_hidden)
            self.position_encoding = get_2d_sincos_pos_embed(embed_dim=self.d_hidden, grid_size=2, cls_token=False).reshape(2, 2, self.d_hidden)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(self.d_hidden, heads, dropout) for _ in range(attention_layers)])

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

        # labels = torch.arange(0, 842).long().to(images.device)  # Replace num_labels with your actual number of labels
        # embeddings = self.label_embeddings(labels)
        #
        # np.save("/home/mila/h/hager.radi/scratch/ecosystem-embedding/rtran_label_embeddings_random.npy", embeddings.detach().cpu().numpy())
        # exit(0)
        z_features = self.backbone(images) # image: HxWxD , out: [128, 4, 512]
        # cls_tokens = self.cls_token.expand(images.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, z_features), dim=1)  # (N, L+1, D)
        # print(x.size())
        # z_features = x

        if self.use_pos_encoding:
            pos_encoding = torch.from_numpy(self.position_encoding).float().to(images.device)
            # print(pos_encoding.size(), z_features.size())
            pos_encoding = pos_encoding.view(1, pos_encoding.size(2), pos_encoding.size(0), pos_encoding.size(1)).repeat(z_features.size(0), 1, 1, 1)
            z_features = z_features + pos_encoding

        z_features = z_features.view(z_features.size(0), z_features.size(1), -1).permute(0, 2, 1)

        if self.use_text_species:
            embedded_species = self.embedded_species.to(images.device)
            embedded_species = torch.transpose(embedded_species, 0, 1) # (2, 670)
            embedded_species = embedded_species.repeat(int(images.size(0)/2), 1) # (128, 670)
            # to handle the last batch:
            if embedded_species.size(0) != images.size(0):
                embedded_species = torch.cat((embedded_species, embedded_species[0:1, :]), 0)
            init_label_embeddings = self.label_embeddings(embedded_species)    # LxD # (128, 670, 512)
        else:
            const_label_input = self.label_input.repeat(images.size(0), 1).to(images.device)  # LxD (128, 670)
            init_label_embeddings = self.label_embeddings(const_label_input)    # LxD # (128, 670, 512)

        # print(embedded_species.size(), const_label_input.size(), init_label_embeddings.size())
        # Get state embeddings (mask is 0 or regression value)
        # print(torch.unique(mask))
        # print(self.state_embeddings)
        # unknown_mask = custom_replace(mask, 1, 0, 0)
        mask[mask == -2] = -1
        if self.quantized_mask_bins > 1:
            mask_q[mask_q == -2] = -1
            label_feat_vec = custom_replace_n(mask_q).long()
        else:
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()
        # print(label_feat_vec.unique())
        # if labels is not None:
        #     regression_labels = self.regression_embedding(labels)
        #     init_label_embeddings += (state_embeddings * regression_labels.unsqueeze(-1))
        # else:
        if self.use_lmt:
            # Get state embeddings
            state_embeddings = self.state_embeddings(label_feat_vec)  # (128, 670, 512)
            # print(self.state_embeddings, state_embeddings.size())
            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings
        # concatenate images features to label embeddings
        embeddings = torch.cat((z_features, init_label_embeddings), 1)  # (128, 674, 512)
        # Feed all (image and label) embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        for layer in self.self_attn_layers:
            embeddings = layer(embeddings, mask=None)

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1), device=output.device).unsqueeze(0).repeat(output.size(0), 1, 1)
        output = (output * diag_mask).sum(-1)

        return output

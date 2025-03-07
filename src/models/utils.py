"""
utility functions for R-tran model
"""

import json
import math
from collections import Counter
from datetime import time

import numpy as np
import tifffile as tiff
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

# from gensim.models import KeyedVectors


def load_word2vec_pretrained_weights(word_to_idx, vocab_size, embedding_dim):
    # Path to the downloaded model
    model_path = "/home/mila/h/hager.radi/scratch/ecosystem-embedding/GoogleNews-vectors-negative300.bin.gz"
    # model_path = '/home/mila/h/hager.radi/scratch/ecosystem-embedding/wiki-news-300d-1M-subword.vec.zip'
    # Load the model
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    # Initialize the embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    present_words = 0
    absent_words = 0
    absent_ids = []
    present_ids = []
    for word, idx in word_to_idx.items():
        if word in word2vec_model:
            # Use the Word2Vec embedding if the word is in the model
            present_words += 1
            present_ids.append(idx)
            embedding_matrix[idx] = np.repeat(word2vec_model[word], 2)[:embedding_dim]
        else:
            # Random initialization for words not in the model
            # embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))
            absent_words += 1
            absent_ids.append(idx)

    mean_embeddings = np.mean(embedding_matrix[present_ids], axis=0)
    embedding_matrix[absent_ids] = mean_embeddings
    return embedding_matrix


def tokenize_species(species_file_name):
    with open(species_file_name) as f:
        species_names = [line.rstrip() for line in f]

    # Tokenize
    tokenized_data = [name.lower().split() for name in species_names]

    # Flatten the list and count word frequencies
    word_freq = Counter([word for sp in tokenized_data for word in sp])

    # Create word to index mapping
    word_to_idx = {
        word: i + 1 for i, (word, _) in enumerate(word_freq.items())
    }  # Start indexing from 1
    word_to_idx["<unk>"] = 0  # Add a token for unknown words

    def encode_species(species_name):
        return [
            word_to_idx.get(word, word_to_idx["<unk>"])
            for word in species_name.lower().split()
        ]

    encoded_species = [encode_species(sp) for sp in species_names]

    max_length = max(len(sp) for sp in encoded_species)

    def pad_encoded_sp(encoded_sp):
        return np.pad(encoded_sp, (0, max_length - len(encoded_sp)), mode="constant")

    padded_species = np.array([pad_encoded_sp(shop) for shop in encoded_species])

    vocab_size = len(word_to_idx)

    return padded_species, word_to_idx, vocab_size


def weights_init(module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def custom_replace(tensor, on_neg_1, on_zero, on_one):
    res = tensor.clone()
    res[tensor == -1] = on_neg_1
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res


def custom_replace_n(data: torch.tensor, n: int):
    """
    replacing unique values with their index
    """
    original_values = torch.linspace(1 / n, 1, n)  # Values from 1/n to 1

    original_values = torch.cat((torch.tensor([-1, 0]), original_values), dim=0)
    new_values = torch.arange(0, n + 2, 1)
    res = data.clone()
    for original_value, new_value in zip(original_values, new_values):
        mask = torch.isclose(data, original_value, atol=1e-3)
        res[mask] = new_value

    return res


def masked_loss_custom_replace(tensor, on_neg_2, on_neg_1, on_zero, on_one):
    res = tensor.clone()
    res[tensor == -2] = on_neg_2
    res[tensor == -1] = on_neg_1
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res


def positional_encoding_2d(height, width, d_model):
    assert (
        d_model % 4 == 0
    ), "Dimension of model must be divisible by 4 for 2D positional encoding"

    pos_enc = np.zeros((height, width, d_model))
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    div_term = 10000 ** (np.arange(0, d_model, 4) / d_model)

    pos_enc[:, :, 0::4] = np.sin(x[:, :, None] / div_term)
    pos_enc[:, :, 1::4] = np.cos(x[:, :, None] / div_term)
    pos_enc[:, :, 2::4] = np.sin(y[:, :, None] / div_term)
    pos_enc[:, :, 3::4] = np.cos(y[:, :, None] / div_term)

    return pos_enc


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = emb_sin + emb_cos  # np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        # stop()
        y_embed = not_mask.cumsum(1)  # , dtype=torch.float32)
        x_embed = not_mask.cumsum(2)  # , dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats)  # , dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # stop()

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
    steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


def init_first_layer_weights(
    in_channels: int, rgb_weights, hs_weight_init: str = "random"
):
    """Initializes the weights for filters in the first conv layer.
    If we are using RGB-only, then just initializes var to rgb_weights. Otherwise, uses
    hs_weight_init to determine how to initialize the weights for non-RGB bands.
    Args
    - int: in_channesl, input channels
        - in_channesl is  either 3 (RGB), 7 (lxv3), or 9 (Landsat7) or 2 (NL)
    - rgb_weights: ndarray of np.float32, shape [64, 3, F, F]
    - hs_weight_init: str, one of ['random', 'same', 'samescaled']
    Returs
    -torch tensor : final_weights
    """

    out_channels, rgb_channels, H, W = rgb_weights.shape
    rgb_weights = torch.tensor(rgb_weights, device=rgb_weights.device)
    ms_channels = in_channels - rgb_channels
    if in_channels == 3:
        final_weights = rgb_weights

    elif in_channels < 3:
        with torch.no_grad():
            mean = rgb_weights.mean()
            std = rgb_weights.std()
            final_weights = torch.empty(
                (out_channels, in_channels, H, W), device=rgb_weights.device
            )
            final_weights = torch.nn.init.trunc_normal_(final_weights, mean, std)
    elif in_channels > 3:
        # spectral images

        if hs_weight_init == "same":

            with torch.no_grad():
                mean = rgb_weights.mean(
                    dim=1, keepdim=True
                )  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = mean

        elif hs_weight_init == "random":
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                ms_weights = torch.empty(
                    (out_channels, ms_channels, H, W), device=rgb_weights.device
                )
                ms_weights = torch.nn.init.trunc_normal_(ms_weights, mean, std)
            print(f"random: {time.time() - start}")

        elif hs_weight_init == "samescaled":
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean(
                    dim=1, keepdim=True
                )  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = (mean * 3) / (3 + ms_channels)
                # scale both rgb_weights and ms_weights
                rgb_weights = (rgb_weights * 3) / (3 + ms_channels)

        else:

            raise ValueError(f"Unknown hs_weight_init type: {hs_weight_init}")

        final_weights = torch.cat([rgb_weights, ms_weights], dim=1)
    return final_weights


def load_geotiff_visual(file):
    img = tiff.imread(file).astype(np.float32)

    img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

    return img


def load_geotiff(file):
    img = tiff.imread(file)
    new_band_order = [2, 1, 0, 3]  # r, g, b, nir
    img = img[:, :, new_band_order].astype(float)
    img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

    return img


def json_load(file_path):
    """
    loads a json file given path
    """
    with open(file_path, "r") as f:
        return json.load(f)

#
#
# if __name__ == "__main__":
#     x = torch.tensor([-1.0000, -1.0000,  0.1667,  1.0000,  0.5000,  1.0000,  0.0000])
#     print(custom_replace_n(x, 6))
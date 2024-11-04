"""
This file includes label (target) masking for RTran training and evaluation.
"""

import os
import random

import numpy as np
import torch


def multi_species_masking(per_taxa_species_count: list[int]):
    """
    function for handling species masking when all species set is available in a data sample
    Parameters:
        species set [List]: a list of species sizes [bird species, butterfly species]. When predicting the full checklist of birds and butterflies,
        there is an assumption that targets are ordered as [bird species, butterfly species]. If species set= None, we are working on SatBird only
        num_labels: Total number of labels
        max_known: probability ratio (between 0 and 1) for known labels
    Returns:
        unk_mask_indices [List]: list of unknown indices
    """
    # when all species (birds and butterflies) are there
    species_index_to_mask = np.random.randint(0, 2)
    if species_index_to_mask == 0:  # mask all birds (unknown birds)
        unk_mask_indices = np.arange(0, per_taxa_species_count[0])
    else:  # mask all butterflies (unknown butterflies)
        unk_mask_indices = np.arange(
            per_taxa_species_count[0],
            per_taxa_species_count[0] + per_taxa_species_count[1],
        )

    return unk_mask_indices


def random_species_masking(available_species_mask, max_known):
    non_zero_indices = torch.nonzero(available_species_mask, as_tuple=False).flatten()
    num_labels = len(non_zero_indices)
    num_known = random.randint(
        0, int(num_labels * max_known)
    )  # known: 0, 0.75l -> unknown: l, 0.25l
    shuffled_indices = non_zero_indices[torch.randperm(len(non_zero_indices))]
    unk_mask_indices = shuffled_indices[: int(num_labels - num_known)]

    return unk_mask_indices


def songbird_masking(index, data_base_dir):
    """
    masking songbirds or nonsongbirds given an index
    index 0: non-songbird
    index 1: songbirds
    """
    songbird_indices = [
        "nonsongbird_indices.npy",
        "songbird_indices.npy",
    ]
    unk_mask_indices = np.load(os.path.join(data_base_dir, songbird_indices[index]))
    return unk_mask_indices


def bird_species_masking(
    per_taxa_species_count: list[int],
    max_known: float,
    available_species_index: int,
    data_base_dir: str,
) -> torch.Tensor:
    """
    masking butterfly species given per_taxa_species_count and max_known
    """
    if random.random() < 0.5:  # mask songbirds vs. nonsongbirds with this probability
        set_to_mask = np.random.randint(0, 2)
        unk_mask_indices = songbird_masking(set_to_mask, data_base_dir)
    else:
        num_known = random.randint(
            0, int(per_taxa_species_count[available_species_index] * max_known)
        )
        unk_mask_indices = random.sample(
            list(np.arange(0, per_taxa_species_count[0])),
            int(per_taxa_species_count[available_species_index] - num_known),
        )

    return unk_mask_indices


def butterfly_species_masking(
    per_taxa_species_count: list[int], max_known: float, available_species_index: int
) -> torch.Tensor:
    """
    masking butterfly species given per_taxa_species_count and max_known
    """
    num_known = random.randint(
        0, int(per_taxa_species_count[available_species_index] * max_known)
    )
    unk_mask_indices = random.sample(
        list(
            np.arange(
                per_taxa_species_count[0],
                per_taxa_species_count[0] + per_taxa_species_count[1],
            )
        ),
        int(per_taxa_species_count[available_species_index] - num_known),
    )

    return unk_mask_indices


# TODO: replace all numpy operations with torch
def get_unknown_mask_indices(
    num_labels,
    mode,
    available_species_mask,
    max_known=0.5,
    multi_taxa=False,
    per_taxa_species_count=None,
    predict_family_of_species=-1,
    data_base_dir=None,
):
    """
    sample random number of known labels during training
    num_labels: total number of species
    mode: train, val or test
    max_known: number of known values at max
    absent species: if not -1, birds are absent (0), butterflies are absent (1)
    species set: a list of species sizes [bird species, butterfly species]
    predict family of species: (only if not training) mask out either birds or butterflies
    data_base_dir: root directory for data, used to access songbird and nonsongbird indices
    """
    # true when SatBird is available for multi_taxa, or true for SatBird only
    random.seed()
    if mode == "train":
        available_species_index = -1
        if multi_taxa:
            if torch.count_nonzero(available_species_mask) == per_taxa_species_count[0]:
                available_species_index = 0
            elif (
                torch.count_nonzero(available_species_mask) == per_taxa_species_count[1]
            ):
                available_species_index = 1

        if available_species_index == -1 and multi_taxa:  # all species are there
            #unk_mask_indices = random_species_masking(
            #available_species_mask=available_species_mask, max_known=max_known
        #)
            unk_mask_indices = multi_species_masking(
                per_taxa_species_count=per_taxa_species_count,
            )
        elif (available_species_index == 0 and multi_taxa) or (
            available_species_index == -1 and not multi_taxa
        ):  # butterflies missing for multi-taxa or SatBird only for non multi-taxa
            unk_mask_indices = bird_species_masking(
                per_taxa_species_count=per_taxa_species_count,
                max_known=max_known,
                available_species_index=available_species_index,
                data_base_dir=data_base_dir,
            )
        elif (
            available_species_index == 1 and multi_taxa
        ):  # birds missing for multi-taxa
            unk_mask_indices = butterfly_species_masking(
                per_taxa_species_count=per_taxa_species_count,
                max_known=max_known,
                available_species_index=available_species_index,
            )

    elif mode == "val":  # random unknown indices over available species
        unk_mask_indices = random_species_masking(
            available_species_mask=available_species_mask, max_known=max_known
        )

    elif mode == "test":
        if (
            predict_family_of_species == 1 and multi_taxa
        ):  # butterflies to eval in multi taxa setup
            unk_mask_indices = np.arange(
                per_taxa_species_count[0],
                per_taxa_species_count[0] + per_taxa_species_count[1],
            )
        elif (
            predict_family_of_species == 0 and multi_taxa
        ):  # birds to eval in multi taxa setup
            unk_mask_indices = np.arange(0, per_taxa_species_count[0])
        elif (
            predict_family_of_species != -1 and not multi_taxa
        ):  # non-songbirds / songbirdss to eval in SatBird only setup
            unk_mask_indices = songbird_masking(
                index=predict_family_of_species, data_base_dir=data_base_dir
            )
        else:  # random unknown indices over all available species
            unk_mask_indices = random_species_masking(
                available_species_mask=available_species_mask, max_known=max_known
            )

    return unk_mask_indices

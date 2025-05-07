"""
This file includes label (target) masking for CISO training and evaluation.
"""

import random

import numpy as np
import torch


def random_species_masking(available_species_mask, max_known):
    non_zero_indices = torch.nonzero(available_species_mask, as_tuple=False).flatten()
    num_labels = len(non_zero_indices)
    num_known = random.randint(
        0, int(num_labels * max_known)
    )  # known: 0, 0.75l -> unknown: l, 0.25l
    shuffled_indices = non_zero_indices[torch.randperm(len(non_zero_indices))]
    unk_mask_indices = shuffled_indices[: int(num_labels - num_known)]

    return unk_mask_indices


def single_taxa_species_masking(main_taxa_dataset_name: str, index: int, species_list_masked: list):
    def trees_masking(index: int, species_list_masked: list):
        """
        masking trees or not-trees given an index
        index 0: isTree=False
        index 1: isTree=True
        """
        return np.where(species_list_masked["isTree"] == bool(index))[0]

    def songbird_masking(index: int, species_list_masked: list):
        """
        masking songbirds or nonsongbirds given an index
        index 0: isSongbird = False
        index 1: isSongbird = True
        """
        return np.where(species_list_masked == bool(index))[0]

    if main_taxa_dataset_name == "splot":
        return trees_masking(index=index, species_list_masked=species_list_masked)
    elif main_taxa_dataset_name == "satbird":
        return songbird_masking(index=index, species_list_masked=species_list_masked)

    return None


# TODO: replace all numpy operations with torch
def get_unknown_mask_indices(
    mode,
    available_species_mask,
    species_list_masked,
    max_known=0.5,
    multi_taxa=False,
    per_taxa_species_count=None,
    predict_family_of_species=-1,
    main_taxa_dataset_name="satbird",
):
    """
    sample random number of known labels during training
    num_labels: total number of species
    mode: train, val or test
    max_known: number of known values at max
    species_list_masked: list of species by their taxa (tree vs. non-tree or songbird vs. nonsongbird)
    absent species: if not -1, birds are absent (0), butterflies are absent (1)
    species set: a list of species sizes [bird species, butterfly species]
    predict family of species: (only if not training) mask out either birds or butterflies
    data_base_dir: root directory for data, used to access songbird and nonsongbird indices
    """
    # true when SatBird is available for multi_taxa, or true for SatBird only
    random.seed()
    if mode in ["train", "val"]:
        unk_mask_indices = random_species_masking(
            available_species_mask=available_species_mask, max_known=max_known
        )

    elif mode == "test":
        if predict_family_of_species != -1 and multi_taxa:
            taxa_indices = {
                # birds (multi_taxa index 0) to eval in multi taxa setup
                0: np.arange(0, list(per_taxa_species_count.values())[0]),  # birds
                # butterflies or trees (multi_taxa index 1) to eval in multi taxa setup
                1: np.arange(list(per_taxa_species_count.values())[0],
                             list(per_taxa_species_count.values())[0] + list(per_taxa_species_count.values())[1])
            }
            unk_mask_indices = taxa_indices.get(predict_family_of_species)
        elif (
            predict_family_of_species != -1 and not multi_taxa
        ):  # non-songbirds / songbirds to eval in SatBird only setup | non-trees / trees to eval in sPlots setup
            unk_mask_indices = single_taxa_species_masking(
                index=predict_family_of_species, species_list_masked=species_list_masked, main_taxa_dataset_name=main_taxa_dataset_name
            )
        else:  # random unknown indices over all available species
            unk_mask_indices = random_species_masking(
                available_species_mask=available_species_mask, max_known=max_known
            )

    return unk_mask_indices

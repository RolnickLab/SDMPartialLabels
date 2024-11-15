"""
This file includes label (target) masking for RTran training and evaluation.
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
        return np.where(species_list_masked["isSongbird"] == bool(index))[0]

    if main_taxa_dataset_name == "splot":
        return trees_masking(index=index, species_list_masked=species_list_masked)
    elif main_taxa_dataset_name == "satbird":
        return songbird_masking(index=index, species_list_masked=species_list_masked)

    return None


def taxon_species_masking(
    available_species_mask: list[int],
    max_known: float,
    main_taxa_dataset_name: str,
    species_list_masked: list,
) -> torch.Tensor:
    """
    masking butterfly species given per_taxa_species_count and max_known
    """
    if random.random() < 0.5:  # mask songbirds vs. nonsongbirds with this probability
        set_to_mask = np.random.randint(0, 2)
        unk_mask_indices = single_taxa_species_masking(main_taxa_dataset_name, set_to_mask, species_list_masked)
    else:
        unk_mask_indices = random_species_masking(available_species_mask, max_known)

    return unk_mask_indices


def multi_taxa_species_masking(per_taxa_species_count: list[int]):
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
        if multi_taxa:
            taxa_indices = {
                # butterflies or trees (multi_taxa index 1) to eval in multi taxa setup
                1: np.arange(per_taxa_species_count.values()[0], per_taxa_species_count.values()[0] + per_taxa_species_count.values()[1]),
                # birds (multi_taxa index 0) to eval in multi taxa setup
                0: np.arange(0, per_taxa_species_count.values()[0])  # birds
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

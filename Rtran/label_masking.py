"""
This file includes label (target) masking for RTran training and evaluation.
"""

import os
import random

import numpy as np


def multi_species_masking(species_set, num_labels, max_known, data_base_dir):
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
    # when all species (birds and butterflies)| are there
    if (species_set is not None and
        len(species_set)> 1 and random.random() < 0.5
    ):  # 50 % of the time, mask either birds or butterflies
        # assume one of them is absent # 0 (birds) or 1 (butterflies)
        absent_species = np.random.randint(0, 2)
        if absent_species == 0:  # mask all birds (unknown birds)
            unk_mask_indices = np.arange(0, species_set[0])
        else:  # mask all butterflies (unknown butterflies)
            unk_mask_indices = np.arange(
                species_set[0], species_set[0] + species_set[1]
            )
    # For SatBird only, 50% of the time, mask either songbirds or non-songbirds
    elif (
        species_set is None and random.random() < 0.5
    ):  # mask songbirds vs. nonsongbirds with this probability
        set_to_mask = np.random.randint(0, 2)
        unk_mask_indices = songbird_masking(set_to_mask, data_base_dir)
    
    else:  # mask randomly from all species
        num_known = random.randint(
            0, int(num_labels * max_known)
        )  # known: 0, 0.75l -> unknown: l, 0.25l
        unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))

    return unk_mask_indices


def songbird_masking(index, data_base_dir):
    """
    masking songbirds or nonsongbirds given an index
    index 0: non-songbird
    index 1: songbirds
    """
    songbird_indices = [
        "stats/nonsongbird_indices.npy",
        "stats/songbird_indices.npy",
    ]
    unk_mask_indices = np.load(os.path.join(data_base_dir, songbird_indices[index]))
    return unk_mask_indices


def bird_species_masking(species_set, max_known):
    """
    masking bird species
    """
    present_species_index = 0

    num_known = random.randint(0, int(species_set[present_species_index] * max_known))
    unk_mask_indices = random.sample(
        list(np.arange(0, species_set[0])),
        int(species_set[present_species_index] - num_known),
    )

    return unk_mask_indices


def butterfly_species_masking(species_set, max_known):
    """
    masking butterfly data given species set and max_known
    """
    present_species_index = 1

    num_known = random.randint(0, int(species_set[present_species_index] * max_known))
    unk_mask_indices = random.sample(
        list(np.arange(species_set[0], species_set[0] + species_set[1])),
        int(species_set[present_species_index] - num_known),
    )

    return unk_mask_indices


def get_unknown_mask_indices(
    num_labels,
    mode,
    max_known=0.5,
    absent_species=-1,
    species_set=None,
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
    
    if mode == "train":
        random.seed()
        if absent_species == -1:  # all species are there
            unk_mask_indices = multi_species_masking(
                species_set=species_set,
                num_labels=num_labels,
                max_known=max_known,
                data_base_dir=data_base_dir,
            )
        elif absent_species == 1:  # butterflies missing
            unk_mask_indices = bird_species_masking(
                species_set=species_set, max_known=max_known
            )
        elif absent_species == 0:  # birds missing
            unk_mask_indices = butterfly_species_masking(
                species_set=species_set, max_known=max_known
            )
    else:
        if predict_family_of_species == 1:
            # to predict butterflies only (or songbird only if only birds data are there)
            if species_set is None:
                unk_mask_indices = songbird_masking(
                    index=predict_family_of_species, data_base_dir=data_base_dir
                )
            elif species_set == "all":
                unk_mask_indices = np.arange(num_labels)
            else:
                unk_mask_indices = np.arange(
                    species_set[0], species_set[0] + species_set[1]
                )
        elif predict_family_of_species == 0:
            # to predict birds only (or non-songbirds only if only birds data are there)
            if species_set is None:
                unk_mask_indices = songbird_masking(
                    index=predict_family_of_species, data_base_dir=data_base_dir
                )
            elif species_set == "all":
                unk_mask_indices = np.arange(num_labels)
            else:
                unk_mask_indices = np.arange(0, species_set[0])
        else:  # when no certain family of species is specified
            if absent_species == 1:  # butterflies missing
                unk_mask_indices = bird_species_masking(
                    species_set=species_set, max_known=max_known
                )
            elif absent_species == 0:  # birds missing
                unk_mask_indices = butterfly_species_masking(
                    species_set=species_set, max_known=max_known
                )
            else:
                num_known = int(num_labels * max_known)
                unk_mask_indices = random.sample(
                    range(num_labels), int(num_labels - num_known)
                )

    return unk_mask_indices

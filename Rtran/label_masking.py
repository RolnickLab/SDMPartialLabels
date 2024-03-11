import json
import numpy as np
import random


def multi_species_masking(species_set, num_labels, max_known):
    """
    function for handling species masking when all species set is available in a data sample
    Parameters:
        species set [List]: a list of species sizes [bird species, butterfly species]
        num_labels: Total number of labels
        max_known: probability ratio (between 0 and 1) for known labels
    Returns:
        unk_mask_indices [List]: list of unknown indices
    """
    songbird_indices = ["/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/nonsongbird_indices.npy",
     "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/songbird_indices.npy"]

    # when all species (birds and butterflies)| are there
    if species_set is not None and random.random() < 0.5: # 50 % of the time, mask either birds or butterflies
        absent_species = int(np.random.randint(0, 2, 1)[0])  # 0 or 1
        present_species = 1 - absent_species
        if absent_species == 0: # mask all butterflies (unknown butterflies)
            unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                         species_set[absent_species] + (present_species * species_set[present_species]))
        else: # mask all birds (unknown birds)
            unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                         species_set[present_species] + (
                                                     present_species * species_set[present_species]))
    # for SatBird only, 50% of the time, mask either songbirds or non-songbirds
    elif species_set is None and random.random() < 0.5: # mask songbirds vs. nonsongbirds with this probability
            set_to_mask = int(np.random.randint(0, 2, 1)[0])
            unk_mask_indices = np.load(songbird_indices[set_to_mask])
    else: # mask randomly from all species
        num_known = random.randint(0, int(num_labels * max_known))  # known: 0, 0.75l -> unknown: l, 0.25l
        unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))

    return unk_mask_indices


def songbird_masking(index):
    """
    masking songbirds or nonsongbirds given an index
    index 0: non-songbird
    index 1: songbirds
    """
    songbird_indices = ["/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/nonsongbird_indices.npy",
     "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/songbird_indices.npy"]
    unk_mask_indices = np.load(songbird_indices[index])
    return unk_mask_indices


def bird_species_masking(species_set, max_known):
    """
    masking bird species
    """
    present_species_index = 0

    num_known = random.randint(0, int(species_set[present_species_index] * max_known))
    unk_mask_indices = random.sample(list(np.arange(0, species_set[present_species_index])),
                                     int(species_set[present_species_index] - num_known))

    return unk_mask_indices


def butterfly_species_masking(species_set, max_known):
    """
    masking butterfly data given species set and max_known
    """
    present_species_index = 1

    num_known = random.randint(0, int(species_set[present_species_index] * max_known))
    unk_mask_indices = random.sample(list(np.arange(species_set[(1 - present_species_index)],
                                                    species_set[(1 - present_species_index)] + species_set[present_species_index])),
                                     int(species_set[present_species_index] - num_known))

    return unk_mask_indices


def get_unknown_mask_indices(num_labels, mode, max_known=0.5, absent_species=-1,
                             species_set=None, predict_family_of_species=-1):
    """
    sample random number of known labels during training
    num_labels: total number of species
    mode: train, val or test
    max_known: number of known values at max
    absent species: if not -1, birds are absent (0), butterflies are absent (1)
    species set: a list of species sizes [bird species, butterfly species]
    predict family of species: (only if not training) mask out either birds or butterflies
    """

    if mode == 'train':
        random.seed()
        if absent_species == -1: # all species are there
            unk_mask_indices = multi_species_masking(species_set=species_set, num_labels=num_labels, max_known=max_known)
        elif absent_species == 1: # butterflies missing
            unk_mask_indices = bird_species_masking(species_set=species_set, max_known=max_known)
        elif absent_species == 0: #birds missing
            unk_mask_indices = butterfly_species_masking(species_set=species_set, max_known=max_known)
    else:
        # for validation or testing
        if predict_family_of_species == 1:
            # to predict butterflies only (or songbird only if only birds data are there)
            if species_set is None:
                unk_mask_indices = songbird_masking(predict_family_of_species)
            else:
                unk_mask_indices = np.arange(species_set[0], species_set[0] + species_set[1])
        elif predict_family_of_species == 0:
            # to predict birds only (or non-songbirds only if only birds data are there)
            if species_set is None:
                unk_mask_indices = songbird_masking(predict_family_of_species)
            else:
                unk_mask_indices = np.arange(0, species_set[0])
        else: # when no certain family of species is specified
            if absent_species == 1: # butterflies missing
                unk_mask_indices = bird_species_masking(species_set=species_set, max_known=max_known)
            elif absent_species == 0: #birds missing
                unk_mask_indices = butterfly_species_masking(species_set=species_set, max_known=max_known)
            else:
                num_known = int(num_labels * max_known)
                unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))

    return unk_mask_indices

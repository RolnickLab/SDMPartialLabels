import json
import numpy as np
import random


def multi_species_masking(species_set, num_labels, max_known):
    per_species_mask_file = "/network/projects/ecosystem-embeddings/SatButterfly_v2/USA/bird_species_family_mapping.json"
    songbird_indices = ["/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/nonsongbird_indices.npy",
     "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/songbird_indices.npy"]

    with open(per_species_mask_file, 'r') as f:
        per_species_mask = json.load(f)
        mask_max_size = len(per_species_mask.keys())

    # when all species are there
    if species_set is not None and random.random() < 0.5:
        absent_species = int(np.random.randint(0, 2, 1)[0])  # 0 or 1
        present_species = 1 - absent_species
        if absent_species == 0:
            unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                         species_set[absent_species] + (present_species * species_set[present_species]))
        else:
            unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                         species_set[present_species] + (
                                                     present_species * species_set[present_species]))
    else:
        # if random.random() < 0.6: # mask songbirds vs. nonsongbirds with this probability
        #     set_to_mask = int(np.random.randint(0, 2, 1)[0])
        #     unk_mask_indices = np.load(songbird_indices[set_to_mask])
        # else:
        # absent_species = int(np.random.randint(0, mask_max_size, 1)[0])
        # unk_mask_indices = np.array(list(per_species_mask.values())[absent_species])
        num_known = random.randint(0, int(num_labels * max_known))  # known: 0, 0.75l -> unknown: l, 0.25l
        unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))

    return unk_mask_indices


def songbird_masking(index):
    """
    index 0: non-songbird
    index 1: songbirds
    """
    songbird_indices = ["/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/nonsongbird_indices.npy",
     "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/songbird_indices.npy"]
    unk_mask_indices = np.load(songbird_indices[index])
    return unk_mask_indices


def bird_species_masking(absent_species, species_set, max_known):
    present_species = 1 - absent_species

    # what_to_mask = int(np.random.randint(0, mask_max_size, 1)[0])
    # unk_mask_indices = np.array(list(per_species_mask.values())[what_to_mask])
    num_known = random.randint(0, int(species_set[present_species] * max_known))  # known: 0, 0.75l -> unknown: l, 0.25l
    unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                    species_set[present_species] + (
                                                            present_species * species_set[
                                                        absent_species]))),
                                     int(species_set[present_species] - num_known))

    return unk_mask_indices


def butterfly_species_masking(absent_species, species_set, max_known):
    present_species = 1 - absent_species
    num_known = random.randint(0, int(species_set[present_species] * max_known))  # known: 0, 0.75l -> unknown: l, 0.25l
    unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                    species_set[absent_species] + (
                                                            present_species * species_set[
                                                        present_species]))),
                                     int(species_set[present_species] - num_known))

    return unk_mask_indices


def get_unknown_mask_indices(num_labels, mode, max_known=0.5, absent_species=-1,
                             species_set=None, predict_family_of_species=-1):
    """
    num_labels: total number of species
    mode: train, val or test
    max_unknown: number of unknown values at max
    absent species: if not -1, birds are absent (0), butterflies are absent (1)
    species set: a list of [bird species, butterfly species]
    predict family of species: (only if not training) mask out either birds or butterflies
    """
    # sample random number of known labels during training; in testing, everything is unknown

    if mode == 'train': # all species are there
        random.seed()
        if absent_species == -1:  # 50% of the time when butterflies are there, mask all butterflies
            unk_mask_indices = multi_species_masking(species_set=species_set, num_labels=num_labels, max_known=max_known)
        elif absent_species == 1: # butterflies missing
            unk_mask_indices = bird_species_masking(absent_species=absent_species, species_set=species_set, max_known=max_known)
        elif absent_species == 0: #birds missing
            unk_mask_indices = butterfly_species_masking(absent_species=absent_species, species_set=species_set, max_known=max_known)
    else:
        # for testing, everything is unknown
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
        else:
            if absent_species == 1: # butterflies missing
                unk_mask_indices = bird_species_masking(absent_species=absent_species, species_set=species_set, max_known=max_known)
            elif absent_species == 0: #birds missing
                unk_mask_indices = butterfly_species_masking(absent_species=absent_species, species_set=species_set, max_known=max_known)
            else:
                num_known = int(num_labels * max_known)
                unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))

    return unk_mask_indices

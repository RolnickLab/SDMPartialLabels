"""
utility functions to get information about species and their frequencies
"""
import json
import os.path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_species_frequencies(root_dir, output_file="species_frequencies.npy", df_file_name="all_summer_hotspots.csv", num_species=670):
    """
    computes frequency of all species given a csv of hotspots
    """
    df = pd.read_csv(os.path.join(root_dir, df_file_name))

    list_of_frequencies = np.zeros(num_species)

    for i, row in tqdm(df.iterrows()):
        species = json.load(open(os.path.join(root_dir, "corrected_targets", row["hotspot_id"] + '.json')))
        species_probs = np.array(species["probs"])

        species_probs = species_probs * row["num_complete_checklists"]
        list_of_frequencies += species_probs

    print(list_of_frequencies)
    np.save(os.path.join(root_dir, output_file), np.array(list_of_frequencies))


def get_subset_of_species(list_of_frequencies, n=20, output_file="random_subset.npy"):
    """
    get a subset of species given n
    must include the least common species and most common species
    """
    most_common_index = get_most_common_species(list_of_frequencies)
    least_common_index = get_least_common_species(list_of_frequencies)

    list_of_species = np.zeros(n)
    list_of_species[0] = most_common_index
    list_of_species[1] = least_common_index

    list_of_species[2:] = np.random.choice(np.nonzero(list_of_frequencies)[0], n - 2, replace=False)

    np.save(os.path.join(root_dir, output_file), list_of_species)


def get_most_common_species(species_frequencies, output_file=None):
    """
    return index of most common species
    """
    index = np.argmax(species_frequencies)
    if output_file:
        np.save(os.path.join(root_dir, output_file), np.array([index]))
    return index


def get_least_common_species(species_frequencies, output_file=None):
    """
    return index of least_common_species
    """
    index = np.argmin(species_frequencies[np.nonzero(species_frequencies)])
    if output_file:
        np.save(os.path.join(root_dir, output_file), np.array([index]))
    return index


def find_zero_occurance_species(root_dir, summer_species='USA_summer/species_frequencies_updated.npy',
                                winter_species='USA_winter/species_frequencies_updated.npy',
                                outfile='USA_summer/missing_species_updated.npy'):
    summer_freq = np.load(os.path.join(root_dir, summer_species))
    winter_freq = np.load(os.path.join(root_dir, winter_species))

    summer_zero_indices = np.where(summer_freq == 0)[0]
    print("Species with 0 frequency in summer: ", len(summer_zero_indices))
    winter_zero_indices = np.where(winter_freq == 0)[0]
    print("Species with 0 frequency in winter: ", len(winter_zero_indices))
    missing_species = np.intersect1d(winter_zero_indices, summer_zero_indices)
    print("Number of missing species: ", len(missing_species))
    np.save(os.path.join(root_dir, outfile), missing_species)


def classify_ebird_species(root_dir, species_file_name, original_checklist_file_name):
    # classify ebird species names into songbirds and non-songbirds
    ebird_species = open(os.path.join(root_dir, species_file_name)).read().split("\n")[:-1]
    ABA_checklist = pd.read_csv(os.path.join(root_dir, original_checklist_file_name))

    # everything below (and including) Masked Tityra is a songbird, everything before is not
    row_threshold = ABA_checklist.index[ABA_checklist['name_1'] == "Masked Tityra"][0]

    # I had to do these manually from checklist 8.0.8 as they don't exist in 8.12
    missing_dict = {"Accipiter gentilis": 0, "Cistothorus platensis": 1, "Empidonax occidentalis": 1, "Leucolia violiceps": 0}
    songbird_classification = []

    for ebird_sp in ebird_species:
        row_found = ABA_checklist.index[ABA_checklist['name_3'] == ebird_sp]
        if len(row_found) > 0:
            row_index = row_found[0]
            if row_index >= row_threshold:
                songbird_classification.append(1)
            else:
                songbird_classification.append(0)
        else:
            songbird_classification.append(missing_dict[ebird_sp])

    songbird_classification = np.array(songbird_classification)

    d = {'species_name': ebird_species, 'songbird_classification': songbird_classification}
    new_df = pd.DataFrame(data=d)
    new_df.to_csv(os.path.join(root_dir, "species_list_with_songbird_classification.csv"))

    np.save(os.path.join(root_dir, "songbird_indices.npy"), np.where(songbird_classification == 1)[0])
    np.save(os.path.join(root_dir, "nonsongbird_indices.npy"), np.where(songbird_classification == 0)[0])


if __name__ == "__main__":
    root_dir = "/network/projects/ecosystem-embeddings/SatBird_data_v2"
    species_freq_file_name = "species_frequencies_updated.npy"
    # compute_species_frequencies(root_dir, species_freq_file_name)
    # find_zero_occurance_species(root_dir)
    classify_ebird_species(root_dir="../../species_data/",
                        species_file_name="species_list_USA.txt",
                        original_checklist_file_name="ABA_Checklist-8.12_mod.csv")
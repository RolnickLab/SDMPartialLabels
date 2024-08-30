import csv
import os

import numpy as np
import pandas as pd


def multi_label_accuracy(y_pred, y_true, threshold=0.5):
    y_pred_binary = (y_pred > threshold).float()

    correct_per_label = (y_pred_binary == y_true).sum(dim=0).float()
    accuracy_per_label = correct_per_label / y_true.size(0)  # divide by batch size

    accuracy = accuracy_per_label.mean().item()

    return accuracy


def trees_masking(config):
    if config.predict_family_of_species == -1:
        return None

    targets = np.load(os.path.join(config.base, config.targets))
    species_df = pd.read_csv(os.path.join(config.base, config.species_list))

    species_indices = np.where(
        targets.sum(axis=0) >= config.species_occurrences_threshold
    )[0]

    species_df = species_df.loc[species_indices]
    species_df = species_df.reset_index(drop=True)

    # 0: not trees, 1 : trees
    indices_to_predict = np.where(
        species_df["isTree"] == config.predict_family_of_species
    )[0]

    return indices_to_predict


def save_results_to_csv(results, root_dir, file_name="test_results.csv"):
    output_file = os.path.join(root_dir, file_name)

    with open(output_file, "a+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        csvfile.seek(0)
        if not csvfile.read():
            writer.writeheader()  # Write the header row based on the dictionary keys

        csvfile.seek(0, os.SEEK_END)
        writer.writerow(results)  # Write the values row by row

    print(f"CSV file '{output_file}' has been saved.")
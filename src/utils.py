import os
from os.path import expandvars
from pathlib import Path
from typing import cast
import csv
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def eval_species_split(
    index: int,
    base_data_folder: str,
    multi_taxa: bool,
    per_taxa_species_count: list[int] = None,
) -> np.ndarray:
    if not multi_taxa:
        songbird_indices = [
            "nonsongbird_indices.npy",
            "songbird_indices.npy",
        ]
        indices_to_predict = np.load(
            os.path.join(
                base_data_folder,
                songbird_indices[index],
            )
        )
    else:
        if index == 0:
            indices_to_predict = np.arange(0, per_taxa_species_count[0])
        else:
            indices_to_predict = np.arange(
                per_taxa_species_count[0],
                per_taxa_species_count[0] + per_taxa_species_count[1],
            )

    return indices_to_predict


def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute
    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def multi_label_accuracy(y_pred, y_true, threshold=0.5):
    y_pred_binary = (y_pred > threshold).float()

    correct_per_label = (y_pred_binary == y_true).sum(dim=0).float()
    accuracy_per_label = correct_per_label / y_true.size(0)  # divide by batch size

    accuracy = accuracy_per_label.mean().item()

    return accuracy


def trees_masking(config):
    if config.partial_labels.predict_family_of_species == -1:
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
        species_df["isTree"] == config.partial_labels.predict_family_of_species
    )[0]

    return indices_to_predict

def save_test_results_to_csv(results, root_dir, file_name="test_results.csv"):
    if root_dir is None:
        print("Not saving results")
        return ()
    output_file = os.path.join(root_dir, file_name)

    with open(output_file, "a+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        csvfile.seek(0)
        if not csvfile.read():
            writer.writeheader()  # Write the header row based on the dictionary keys

        csvfile.seek(0, os.SEEK_END)
        writer.writerow(results)  # Write the values row by row

    print(f"CSV file '{output_file}' has been saved.")

def load_opts(path, default, commandline_opts):
    """
    Args:
    path (pathlib.Path): where to find the overriding configuration
        default (pathlib.Path, optional): Where to find the default opts.
        Defaults to None. In which case it is assumed to be a default config
        which needs processing such as setting default values for lambdas and gen
        fields
    """

    if path is None and default is None:
        path = resolve(Path(__file__)).parent.parent / "configs" / "defaults.yaml"
        print(path)
    else:
        print("using config ", path)

    if default is None:
        default_opts = {}
    else:
        print(default)
        if isinstance(default, (str, Path)):
            default_opts = OmegaConf.load(default)
        else:
            default_opts = dict(default)

    if path is None:
        overriding_opts = {}
    else:
        print("using config ", path)
        overriding_opts = OmegaConf.load(path)

    opts = OmegaConf.merge(default_opts, overriding_opts)

    if commandline_opts is not None and isinstance(commandline_opts, dict):
        opts = OmegaConf.merge(opts, commandline_opts)

    conf = cast(DictConfig, opts)
    return conf
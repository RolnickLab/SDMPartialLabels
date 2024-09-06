import os

import numpy as np


def satbird_species_split(index: int, base_data_folder: str) -> np.ndarray:
    songbird_indices = [
        "stats/nonsongbird_indices.npy",
        "stats/songbird_indices.npy",
    ]
    indices_to_predict = np.load(
        os.path.join(
            base_data_folder,
            songbird_indices[index],
        )
    )

    return indices_to_predict
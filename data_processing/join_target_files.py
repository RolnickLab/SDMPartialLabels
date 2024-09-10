import json
import os
import pickle

import numpy as np


def create_dataframe_from_jsons(folder_path):
    all_data = {}

    for json_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, json_file)

        # Ensure we are processing only JSON files
        if json_file.endswith(".json") and os.path.isfile(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)  # Load JSON data as a dictionary
                k = data["hotspot_id"]
                v = np.array(data["probs"])
                all_data[k] = v

    return all_data


index = 0

folder_path = [
    "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/targets",
    "/network/projects/ecosystem-embeddings/SatButterfly_dataset/SatButterfly_v1/USA/butterfly_targets_v1.2",
    "/network/projects/ecosystem-embeddings/SatButterfly_dataset/SatButterfly_v2/USA/butterfly_targets_v1.2",
]

out_file = [
    "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/satbird_usa_summer_targets.pkl",
    "/network/projects/ecosystem-embeddings/SatButterfly_dataset/SatButterfly_v1/USA/butterfly_all_targets_v1.2.pkl",
    "/network/projects/ecosystem-embeddings/SatButterfly_dataset/SatButterfly_v2/USA/butterfly_all_targets_v1.2.pkl",
]

data_dict = create_dataframe_from_jsons(folder_path[index])

with open(out_file[index], "wb") as pickle_file:
    pickle.dump(data_dict, pickle_file)

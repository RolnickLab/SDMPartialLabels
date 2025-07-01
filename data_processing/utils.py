# create a single pickle target file for SatBird and Satbutterfly
import json
import os
import pickle

import numpy as np


def merge_satbutterfly_targets(src_file_1: str, src_file_2: str, dst_file: str):
    """
    merge two pickle files into one
    :return:
    """
    # Open and load the pickle files
    with open(src_file_1, 'rb') as file1:
        d1 = pickle.load(file1)

    with open(src_file_2, 'rb') as file2:
        d2 = pickle.load(file2)

    # Merge the dictionaries (d1 will be updated with the contents of d2)
    combined_dict = {**d1, **d2}

    # Save the combined dictionary to a new pickle file
    output_file_path = dst_file
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(combined_dict, output_file)

    print(f"Combined dictionary saved to {output_file_path}")


def create_dataframe_from_jsons(folder_path: str):
    """
    create targets dataframe from single json files for satbird/satbutterfly
    """
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


folder_path = [
    "/data/SatBird/USA_summer/targets",
    "/data/SatButterfly/SatButterfly_v1/USA/butterfly_targets_v1.2",
    "/data/SatButterfly/SatButterfly_v2/USA/butterfly_targets_v1.2",
]

out_file = [
    "/data/SatBird/USA_summer/satbird_usa_summer_targets.pkl",
    "/data/SatButterfly/SatButterfly_v1/USA/butterfly_all_targets_v1.2.pkl",
    "/data/network/projects/ecosystem-embeddings/SatButterfly/SatButterfly_v2/USA/butterfly_all_targets_v1.2.pkl",
]

index = 0
data_dict = create_dataframe_from_jsons(folder_path=folder_path[index])
with open(out_file[index], "wb") as pickle_file:
    pickle.dump(data_dict, pickle_file)

# save butterfly v1 and v2 targets in a single pickle file
merge_satbutterfly_targets(src_file_1='/data/SatButterfly/SatButterfly_v1/USA/butterfly_all_targets_v1.2.pkl',
                           src_file_2='/data/SatButterfly/SatButterfly_v2/USA/butterfly_all_targets_v1.2.pkl',
                           dst_file='/data/SatButterfly/combined_SatButterfly_v1Andv2_targets.pkl')

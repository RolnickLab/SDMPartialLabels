import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    root_path = "/Users/hagerradi/Projects/DavidRolnick_collab/Ecosystem_embeddings/sPlots_data"
    world_clim_data = pd.read_csv(os.path.join(root_path, "worldclim_data.csv"))
    print(world_clim_data.columns)
    print(world_clim_data.shape)
    soilgrid_data = pd.read_csv(os.path.join(root_path, "soilgrid_data.csv"))
    print(soilgrid_data.columns)
    print(soilgrid_data.shape)
    print(soilgrid_data['PlotObservationID'].head())
    species_occurrences = np.load(os.path.join(root_path, "species_occurrences-001.npy"))
    print(species_occurrences.shape)
    print(species_occurrences)
    print(np.sum(species_occurrences[60]))
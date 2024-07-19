import os
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, data, targets, species_indices, normalization_means, normalization_stds):
        self.data = data
        self.targets = targets
        self.species_indices = species_indices
        self.normalization_means = normalization_means
        self.normalization_stds = normalization_stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        targets = self.targets[:, self.species_indices]
        targets = targets[idx]

        data = self.data.iloc[idx][self.data.columns[2:]]
        data = (data - self.normalization_means) / (self.normalization_stds + 1e-8)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            targets, dtype=torch.float32
        )


class TabularDataModule(pl.LightningDataModule):
    def __init__(self, data_config, batch_size=32):
        super().__init__()
        self.config = data_config
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        worldclim_df = pd.read_csv(os.path.join(self.config.base, self.config.worldclim_data_path))
        soilgrid_df = pd.read_csv(os.path.join(self.config.base, self.config.soilgrid_data_path))

        data = pd.concat([worldclim_df, soilgrid_df])

        train_split = np.load(os.path.join(self.config.base, self.config.train))
        val_split = np.load(os.path.join(self.config.base, self.config.validation))
        test_split = np.load(os.path.join(self.config.base, self.config.test))
        targets = np.load(os.path.join(self.config.base, self.config.targets))

        species_indices = np.where(targets.sum(axis=0) >= self.config.species_occurrences_threshold)[0]

        self.normalization_means = data[data.columns[2:]].mean()
        self.normalization_stds = data[data.columns[2:]].std()

        self.train_dataset = TabularDataset(
            data=data.iloc[train_split], targets=targets[train_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds
        )
        self.val_dataset = TabularDataset(
            data=data.iloc[val_split], targets=targets[val_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds
        )
        self.test_dataset = TabularDataset(
            data=data.iloc[test_split], targets=targets[test_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

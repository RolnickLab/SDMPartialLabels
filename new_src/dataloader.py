import os
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        species_indices,
        normalization_means,
        normalization_stds,
        df_data_columns,
    ):
        self.data = data
        self.targets = targets
        self.species_indices = species_indices
        self.normalization_means = normalization_means
        self.normalization_stds = normalization_stds
        self.df_data_columns = df_data_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        targets = self.targets[:, self.species_indices]
        targets = targets[idx]

        data = self.data.iloc[idx][self.df_data_columns].tolist()
        data = (data - self.normalization_means) / (self.normalization_stds + 1e-8)

        return torch.tensor(data.tolist(), dtype=torch.float32), torch.tensor(
            targets, dtype=torch.float32
        )


class TabularDataModule(pl.LightningDataModule):
    def __init__(self, data_config, batch_size=32):
        super().__init__()
        self.config = data_config
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        worldclim_df = pd.read_csv(
            os.path.join(self.config.base, self.config.worldclim_data_path)
        )
        soilgrid_df = pd.read_csv(
            os.path.join(self.config.base, self.config.soilgrid_data_path)
        )

        worldclim_env_column_names = [
            "bio_1",
            "bio_2",
            "bio_3",
            "bio_4",
            "bio_5",
            "bio_6",
            "bio_7",
            "bio_8",
            "bio_9",
            "bio_10",
            "bio_11",
            "bio_12",
            "bio_13",
            "bio_14",
            "bio_15",
            "bio_16",
            "bio_17",
            "bio_18",
            "bio_19",
        ]
        soilgrid_env_column_names = [
            "ORCDRC",
            "PHIHOX",
            "CECSOL",
            "BDTICM",
            "CLYPPT",
            "SLTPPT",
            "SNDPPT",
            "BLDFIE",
        ]

        data = pd.concat([worldclim_df, soilgrid_df], axis=1)

        train_split = np.load(os.path.join(self.config.base, self.config.train))
        val_split = np.load(os.path.join(self.config.base, self.config.validation))
        test_split = np.load(os.path.join(self.config.base, self.config.test))
        targets = np.load(os.path.join(self.config.base, self.config.targets))

        species_indices = np.where(
            targets.sum(axis=0) >= self.config.species_occurrences_threshold
        )[0]

        self.normalization_means = data[
            worldclim_env_column_names + soilgrid_env_column_names
        ].mean()
        self.normalization_stds = data[
            worldclim_env_column_names + soilgrid_env_column_names
        ].std()

        self.train_dataset = TabularDataset(
            data=data.iloc[train_split],
            targets=targets[train_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds,
            df_data_columns=worldclim_env_column_names + soilgrid_env_column_names,
        )
        self.val_dataset = TabularDataset(
            data=data.iloc[val_split],
            targets=targets[val_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds,
            df_data_columns=worldclim_env_column_names + soilgrid_env_column_names,
        )
        self.test_dataset = TabularDataset(
            data=data.iloc[test_split],
            targets=targets[test_split],
            species_indices=species_indices,
            normalization_means=self.normalization_means,
            normalization_stds=self.normalization_stds,
            df_data_columns=worldclim_env_column_names + soilgrid_env_column_names,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

import os
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from elapid import MaxentFeatureTransformer
from torch.utils.data import DataLoader, Dataset

from src.dataloaders.label_masking import get_unknown_mask_indices


class sPlotDataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        num_labels,
        species_list_masked=None,
        mode=None,
        maximum_known_labels_ratio=None,
        predict_family=None,
    ):
        self.data = data
        self.targets = targets
        self.hotspots = hotspots

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = {"data": self.data[idx],
                 "targets": self.targets[idx],
                 "hotspot_id": self.hotspots[idx]}
        return batch


class sPlotMaskedDataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        num_labels,
        species_list_masked,
        mode="train",
        predict_family=False,
        maximum_known_labels_ratio=0.75,
    ):
        self.data = data
        self.targets = targets
        self.hotspots = hotspots
        self.num_labels = num_labels
        self.species_list_masked = species_list_masked
        self.mode = mode
        self.predict_family_of_species = predict_family
        self.maximum_known_labels_ratio = maximum_known_labels_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        targets = self.targets[idx]
        data = self.data[idx]
        hotspot = self.hotspots[idx]

        # to exclude species that have no labels
        available_species_mask = (targets != -2).int()

        if self.maximum_known_labels_ratio > 0:
            unk_mask_indices = get_unknown_mask_indices(
                mode=self.mode,
                available_species_mask=available_species_mask,
                max_known=self.maximum_known_labels_ratio,
                predict_family_of_species=self.predict_family_of_species,
                species_list_masked=self.species_list_masked,
                main_taxa_dataset_name="splot"
            )
            mask = targets.clone()
            mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
        else:
            mask = torch.ones_like(targets)
            mask = mask * -1
        batch = {"data": data, "hotspot_id": hotspot, "targets": targets, "mask": mask.long()}
        return batch


class sPlotDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.config = data_config
        self.batch_size = self.config.batch_size
        self.dataloader_to_use = self.config.dataloader_to_use

    def setup(self, stage: Optional[str] = None) -> None:

        worldclim_df = pd.read_csv(
            os.path.join(self.config.base, self.config.worldclim_data_path)
        )
        soilgrid_df = pd.read_csv(
            os.path.join(self.config.base, self.config.soilgrid_data_path)
        )
        species_df = pd.read_csv(
            os.path.join(self.config.base, self.config.species_list)
        )

        df_data_columns = self.config.env_columns

        data = pd.concat([worldclim_df, soilgrid_df], axis=1)
        # remove any duplicate columns after concatenation, example: PlotObservationID
        data = data.loc[:, ~data.T.duplicated()]
        hotspots = data["PlotObservationID"].astype(str).to_list()

        data = data[df_data_columns].to_numpy()

        train_split = np.load(os.path.join(self.config.base, self.config.train))
        val_split = np.load(os.path.join(self.config.base, self.config.validation))
        test_split = np.load(os.path.join(self.config.base, self.config.test))
        targets = np.load(os.path.join(self.config.base, self.config.targets))

        species_indices = np.where(
            targets.sum(axis=0) >= self.config.species_occurrences_threshold
        )[0]

        species_df = species_df.loc[species_indices].reset_index(drop=True)

        # Precompute normalization
        normalization_means = np.mean(data[train_split, :], axis=0)
        normalization_stds = np.std(data[train_split, :], axis=0)

        data = (data - normalization_means) / (normalization_stds + 1e-8)

        train_data = data[train_split]
        val_data = data[val_split]
        test_data = data[test_split]

        if self.config.maxent_transform:
            feature_transformer = MaxentFeatureTransformer(["linear", "hinge", "product", "threshold", "quadratic"])

            train_data = feature_transformer.fit_transform(train_data)
            val_data = feature_transformer.transform(val_data)
            test_data = feature_transformer.transform(test_data)

        self.train_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(train_data, dtype=torch.float32),
            targets=torch.tensor(
                targets[train_split][:, species_indices], dtype=torch.float32
            ),
            hotspots=np.array(hotspots)[train_split],
            species_list_masked=species_df,
            num_labels=len(species_indices),
            mode="train",
            predict_family=self.config.partial_labels.predict_family_of_species,
            maximum_known_labels_ratio=self.config.partial_labels.train_known_ratio,
        )

        self.val_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(val_data, dtype=torch.float32),
            targets=torch.tensor(
                targets[val_split][:, species_indices], dtype=torch.float32
            ),
            hotspots=np.array(hotspots)[val_split],
            species_list_masked=species_df,
            num_labels=len(species_indices),
            mode="val",
            predict_family=self.config.partial_labels.predict_family_of_species,
            maximum_known_labels_ratio=self.config.partial_labels.eval_known_ratio,
        )

        self.test_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(test_data, dtype=torch.float32),
            targets=torch.tensor(
                targets[test_split][:, species_indices], dtype=torch.float32
            ),
            hotspots=np.array(hotspots)[test_split],
            species_list_masked=species_df,
            num_labels=len(species_indices),
            mode="test",
            predict_family=self.config.partial_labels.predict_family_of_species,
            maximum_known_labels_ratio=self.config.partial_labels.eval_known_ratio,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=16,
        )

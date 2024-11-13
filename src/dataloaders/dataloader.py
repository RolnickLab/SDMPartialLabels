import abc
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.dataloaders.label_masking import get_unknown_mask_indices
from src.dataloaders.utils import compute_sampling_weights


class EnvDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and labels at that index
        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: EnvDataset
    size: {len(self)}"""


class SDMEnvDataset(EnvDataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        species_list_masked,
        mode="train",
        maximum_known_labels_ratio=0.5,
        per_taxa_species_count=None,
        multi_taxa=False,
        num_species=670,
        predict_family=-1,
        quantized_mask_bins=1,
    ) -> None:
        """
        this dataloader handles dataset with masks for Ctran model using env variables as inpu
        Parameters:
            data: tensor of input data num_hotspots x env variables
            targets: tensor of targets num_hotspots x num_species,
            mode : train|val|test
            targets_folder: folder name for labels/targets
            maximum_known_labels_ratio: known labels ratio for Ctran
            num_species: total number of species/classes to predict
            per_taxa_species_count: sets of species
            predict_family: -1 for none, 0 if we want to focus on predicting per_taxa_species_count[0], 1 if we want to predict per_taxa_species_count[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """
        super().__init__()
        self.data = data
        self.targets = targets
        self.hotspots = hotspots

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        targets = self.targets[index]
        hotspot_id = self.hotspots[index]

        # to exclude species that have no labels
        available_species_mask = (targets != -2).int()

        return {
            "data": data,
            "targets": targets,
            "hotspot_id": hotspot_id,
            "available_species_mask": available_species_mask,
        }


class SDMEnvMaskedDataset(EnvDataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        species_list_masked,
        mode="train",
        maximum_known_labels_ratio=0.5,
        per_taxa_species_count=None,
        multi_taxa=False,
        num_species=670,
        predict_family=-1,
        quantized_mask_bins=1,
    ) -> None:
        """
        this dataloader handles dataset with masks for Ctran model using env variables as inpu
        Parameters:
            data: tensor of input data num_hotspots x env variables
            targets: tensor of targets num_hotspots x num_species,
            mode : train|val|test
            maximum_known_labels_ratio: known labels ratio for Ctran
            num_species: total number of species/classes to predict
            per_taxa_species_count: sets of species
            predict_family: -1 for none, 0 if we want to focus on predicting per_taxa_species_count[0], 1 if we want to predict per_taxa_species_count[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """
        super().__init__()
        self.data = data
        self.targets = targets
        self.hotspots = hotspots
        self.species_list_masked = species_list_masked
        self.mode = mode
        self.num_species = num_species
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.per_taxa_species_count = per_taxa_species_count
        self.multi_taxa = multi_taxa
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        targets = self.targets[index]
        hotspot_id = self.hotspots[index]

        # to exclude species that have no labels
        available_species_mask = (targets != -2).int()

        mask = targets.clone()

        if self.mode in ["test"] and self.maximum_known_labels_ratio == 0:
            mask = torch.full_like(mask, -1)
        else:
            # constructing known / unknown mask
            unk_mask_indices = get_unknown_mask_indices(
                mode=self.mode,
                max_known=self.maximum_known_labels_ratio,
                available_species_mask=available_species_mask,
                multi_taxa=self.multi_taxa,
                per_taxa_species_count=self.per_taxa_species_count,
                predict_family_of_species=self.predict_family_of_species,
                species_list_masked=self.species_list_masked,
                main_taxa_dataset_name="satbird",
            )
            mask.scatter_(
                dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0
            )

        mask_q = mask.clone()
        mask[mask > 0] = 1

        return {
            "data": data,
            "targets": targets,
            "hotspot_id": hotspot_id,
            "available_species_mask": available_species_mask,
            "mask": mask.long(),
            "mask_q": mask_q,
        }


class SDMDataModule(pl.LightningDataModule):
    """
    SDM - Species Distribution Modeling: works for ebird or ebutterfly
    """

    def __init__(self, opts) -> None:
        super().__init__()
        self.config = opts
        self.seed = self.config.training.seed
        self.batch_size = self.config.data.loaders.batch_size
        self.num_workers = self.config.data.loaders.num_workers
        self.data_base_dir = self.config.data.files.base
        self.targets_file = self.config.data.files.targets_file

        # combining multiple train files
        self.df_train = pd.read_csv(
            os.path.join(self.data_base_dir, self.config.data.files.train[0])
        )
        if len(self.config.data.files.train) > 1:
            for df_file_name in self.config.data.files.train[1:]:
                self.df_train = pd.concat(
                    [
                        self.df_train,
                        pd.read_csv(os.path.join(self.data_base_dir, df_file_name)),
                    ],
                    axis=0,
                )

        # combining multiple validation files
        self.df_val = pd.read_csv(
            os.path.join(self.data_base_dir, self.config.data.files.val[0])
        )
        if len(self.config.data.files.val) > 1:
            for df_file_name in self.config.data.files.val[1:]:
                self.df_val = pd.concat(
                    [
                        self.df_val,
                        pd.read_csv(os.path.join(self.data_base_dir, df_file_name)),
                    ],
                    axis=0,
                )

        # combining multiple testing files
        self.df_test = pd.read_csv(
            os.path.join(self.data_base_dir, self.config.data.files.test[0])
        )
        if len(self.config.data.files.test) > 1:
            for df_file_name in self.config.data.files.test[1:]:
                self.df_test = pd.concat(
                    [
                        self.df_test,
                        pd.read_csv(os.path.join(self.data_base_dir, df_file_name)),
                    ],
                    axis=0,
                )

        self.env = self.config.data.env

        self.predict_family = self.config.predict_family_of_species
        self.num_species = self.config.data.total_species

        # if we are using either SatBird or SatButterfly at a time
        self.dataloader_to_use = self.config.dataloader_to_use

    def get_bird_targets(self, hotspots: list) -> np.array:
        with open(
            os.path.join(self.data_base_dir, self.targets_file[0]), "rb"
        ) as pickle_file:
            data_dict = pickle.load(pickle_file)

        values = [data_dict.get(key, None) for key in hotspots]
        return np.array(values)

    def get_bird_butterfly_targets(self, df, per_taxa_species_count):
        target_files = ["bird", "butterfly", "colocated"]
        target_dict = {}

        for idx, file_key in enumerate(target_files):
            with open(
                os.path.join(self.data_base_dir, self.targets_file[idx]), "rb"
            ) as pickle_file:
                target_dict[file_key] = pickle.load(pickle_file)

        df["species_to_exclude"] = -1  # Initialize to -1 for all species present

        def construct_target(row):
            hotspot_id = row["hotspot_id"]
            target_bird = [-2] * per_taxa_species_count[0]
            target_butterfly = [-2] * per_taxa_species_count[1]

            # Check bird and butterfly presence
            if row["bird"] == 1:
                target_bird = target_dict["bird"].get(hotspot_id, target_bird)

                if row["butterfly"] == 1:
                    target_butterfly = target_dict["colocated"].get(
                        hotspot_id, target_butterfly
                    )

            elif row["butterfly"] == 1:
                target_butterfly = target_dict["butterfly"].get(
                    hotspot_id, target_butterfly
                )
            else:
                raise ValueError(
                    "Cannot have neither butterflies nor birds targets available"
                )

            return list(target_bird) + list(target_butterfly)

        # Construct the target matrix column using `apply`
        df["target"] = df.apply(construct_target, axis=1)
        targets = torch.stack(df["target"].apply(lambda x: torch.Tensor(x)).to_list())
        return targets

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits"""

        train_data = self.df_train[self.env].to_numpy()
        val_data = self.df_val[self.env].to_numpy()
        test_data = self.df_test[self.env].to_numpy()

        train_hotspots = self.df_train["hotspot_id"].tolist()
        val_hotspots = self.df_val["hotspot_id"].tolist()
        test_hotspots = self.df_test["hotspot_id"].tolist()

        normalization_means = np.mean(train_data, axis=0)
        normalization_stds = np.std(train_data, axis=0)

        train_data = (train_data - normalization_means) / (normalization_stds + 1e-8)
        val_data = (val_data - normalization_means) / (normalization_stds + 1e-8)
        test_data = (test_data - normalization_means) / (normalization_stds + 1e-8)

        if self.config.data.multi_taxa:
            train_targets = self.get_bird_butterfly_targets(
                self.df_train, self.config.data.per_taxa_species_count
            )
            val_targets = self.get_bird_butterfly_targets(
                self.df_val, self.config.data.per_taxa_species_count
            )
            test_targets = self.get_bird_butterfly_targets(
                self.df_test, self.config.data.per_taxa_species_count
            )
        else:
            train_targets = self.get_bird_targets(train_hotspots)
            val_targets = self.get_bird_targets(val_hotspots)
            test_targets = self.get_bird_targets(test_hotspots)

        if self.config.data.multi_taxa and self.config.data.loaders.weighted_sampling:
            sample_weights = compute_sampling_weights(
                train_targets, self.config.data.per_taxa_species_count
            )

            # Create a WeightedRandomSampler for balanced sampling
            self.training_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights) * 2,
                replacement=True,
            )
            self.shuffle_training = False
        else:
            self.training_sampler = None
            self.shuffle_training = True

        def get_songbird_indices():
            """To evaluate songbirds vs. non-songbirds"""
            songbird_indices = [
                "nonsongbird_indices.npy",
                "songbird_indices.npy",
            ]
            songbird_indices = np.load(os.path.join(self.data_base_dir, self.config.data.files.satbird_species_indices_path, songbird_indices[1]))
            species_list_masked = np.zeros(self.num_species)
            species_list_masked[songbird_indices] = 1

            return species_list_masked

        self.all_train_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(train_data, dtype=torch.float32),
            targets=torch.tensor(train_targets, dtype=torch.float32),
            hotspots=train_hotspots,
            species_list_masked=get_songbird_indices(),
            mode="train",
            maximum_known_labels_ratio=self.config.partial_labels.train_known_ratio,
            num_species=self.num_species,
            multi_taxa=self.config.data.multi_taxa,
            per_taxa_species_count=self.config.data.per_taxa_species_count,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.partial_labels.quantized_mask_bins,
        )

        self.all_val_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(val_data, dtype=torch.float32),
            targets=torch.tensor(val_targets, dtype=torch.float32),
            hotspots=val_hotspots,
            mode="val",
            species_list_masked=get_songbird_indices(),
            maximum_known_labels_ratio=self.config.partial_labels.eval_known_ratio,
            num_species=self.num_species,
            multi_taxa=self.config.data.multi_taxa,
            per_taxa_species_count=self.config.data.per_taxa_species_count,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.partial_labels.quantized_mask_bins,
        )

        self.all_test_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(test_data, dtype=torch.float32),
            targets=torch.tensor(test_targets, dtype=torch.float32),
            hotspots=test_hotspots,
            species_list_masked=get_songbird_indices(),
            data_base_dir=os.path.join(
                self.data_base_dir, self.config.data.files.satbird_species_indices_path
            ),
            mode="test",
            maximum_known_labels_ratio=self.config.partial_labels.eval_known_ratio,
            num_species=self.num_species,
            multi_taxa=self.config.data.multi_taxa,
            per_taxa_species_count=self.config.data.per_taxa_species_count,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.partial_labels.quantized_mask_bins,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.all_train_dataset,
            batch_size=self.batch_size,
            sampler=self.training_sampler,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=self.shuffle_training,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.all_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.all_test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
        )

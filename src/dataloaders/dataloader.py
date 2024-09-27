import abc
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataloaders.label_masking import get_unknown_mask_indices
from src.models.utils import json_load


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
        exclude,
        hotspots,
        data_base_dir,
        mode="train",
        maximum_known_labels_ratio=0.5,
        species_set=None,
        species_set_eval=None,
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
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            maximum_known_labels_ratio: known labels ratio for Ctran
            num_species: total number of species/classes to predict
            species_set: sets of species
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
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
        species_mask = (targets != -2).int()

        return {
            "data": data,
            "targets": targets,
            "hotspot_id": hotspot_id,
            "species_mask": species_mask,
        }


class SDMEnvCombinedMaskedDataset(EnvDataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        exclude,
        mode="train",
        maximum_known_labels_ratio=0.5,
        num_species=842,
        species_set=None,
        species_set_eval=None,
        predict_family=-1,
        quantized_mask_bins=1,
        data_base_dir=None
    ) -> None:
        """
        this dataloader handles all datasets together SatBird + SatButterfly_v1 + SatButterfly_v2( co-located with some of SatBird positions)
        Parameters:
            data: tensor of input data num_hotspots x env variables
            targets: tensor of targets num_hotspots x num_species,
            exclude: for each hotspot, indicator of species not to consider in the masking proportion because no data is available
            mode : train|val|test
            maximum_known_labels_ratio: known labels ratio for Ctran
            num_species: total number of species/classes to predict
            species_set: set with different species sizes
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """

        super().__init__()
        self.data_base_dir=data_base_dir
        self.data = data
        self.targets = targets
        self.hotspots = hotspots
        self.exclude = exclude
        self.mode = mode
        self.num_species = num_species
        self.species_set = species_set

        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:

        data = self.data[index]
        targets = self.targets[index]
        hotspot_id = self.hotspots[index]
        species_to_exclude = self.exclude[index]

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=self.maximum_known_labels_ratio,
            absent_species=species_to_exclude,
            species_set=self.species_set,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=None
        )

        mask = targets.clone()
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)

        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask

        mask[mask > 0] = 1

        item_ = {
            "data": data,
            "hotspot_id": hotspot_id,
            "targets": targets,
            "mask": mask.long(),
            "eval_mask": mask.long(),
            "mask_q": mask_q.long(),
        }

        return item_


class SDMEnvMaskedDataset(EnvDataset):
    def __init__(
        self,
        data,
        targets,
        hotspots,
        data_base_dir,
        mode="train",
        maximum_known_labels_ratio=0.5,
        species_set=None,
        species_set_eval=None,
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
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            maximum_known_labels_ratio: known labels ratio for Ctran
            num_species: total number of species/classes to predict
            species_set: sets of species
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """
        super().__init__()
        self.data = data
        self.data_base_dir = data_base_dir
        self.targets = targets
        self.hotspots = hotspots
        self.mode = mode
        self.num_species = num_species
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.species_set = species_set
        self.species_set_eval = species_set_eval
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        targets = self.targets[index]
        hotspot_id = self.hotspots[index]
        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=self.maximum_known_labels_ratio,
            species_set=self.species_set,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=self.data_base_dir,
        )

        eval_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=0.0,
            species_set=self.species_set_eval,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=self.data_base_dir,
        )

        mask = targets.clone()
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)

        eval_mask = targets.clone()
        eval_mask.scatter_(
            dim=0, index=torch.Tensor(eval_mask_indices).long(), value=-1.0
        )

        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask

            eval_mask[eval_mask > 0] = 1

        return {
            "data": data,
            "targets": targets,
            "hotspot_id": hotspot_id,
            "mask": mask.long(),
            "eval_mask": mask.long(),
            "mask_q": mask_q.long(),
        }


class SDMDataModule(pl.LightningDataModule):
    """
    SDM - Species Distribution Modeling: works for ebird or ebutterfly
    """

    def __init__(self, opts) -> None:
        super().__init__()
        self.config = opts
        self.seed = self.config.program.seed
        self.batch_size = self.config.data.loaders.batch_size
        self.num_workers = self.config.data.loaders.num_workers
        self.data_base_dir = self.config.data.files.base
        self.targets_file = self.config.data.files.targets_file
        self.target_type = self.config.data.target.type

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

    def get_bird_butterfly_targets(self, df, species_set):
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
            target_bird = [-2] * species_set[0]
            target_butterfly = [-2] * species_set[1]
            species_to_exclude = -1

            # Check bird and butterfly presence
            if row["bird"] == 1:
                target_bird = target_dict["bird"].get(hotspot_id, target_bird)

                if row["butterfly"] == 1:
                    target_butterfly = target_dict["colocated"].get(
                        hotspot_id, target_butterfly
                    )
                else:
                    species_to_exclude = 1  # Only bird present

            elif row["butterfly"] == 1:
                target_butterfly = target_dict["butterfly"].get(
                    hotspot_id, target_butterfly
                )
                species_to_exclude = 0  # Only butterfly present
            else:
                raise ValueError(
                    "Cannot have neither butterflies nor birds targets available"
                )

            return list(target_bird) + list(target_butterfly), species_to_exclude

        # Construct the target matrix and species exclusion column using `apply`
        df["target"], df["species_to_exclude"] = zip(
            *df.apply(construct_target, axis=1)
        )
        targets = torch.stack(df["target"].apply(lambda x: torch.Tensor(x)).to_list())

        return targets, np.array(df["species_to_exclude"]), np.array(df["hotspot_id"])

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

        if self.config.data.species is not None:
            train_targets, exclude_train, train_hotspots = (
                self.get_bird_butterfly_targets(
                    self.df_train, self.config.data.species
                )
            )
            val_targets, exclude_val, val_hotspots = self.get_bird_butterfly_targets(
                self.df_val, self.config.data.species
            )
            test_targets, exclude_test, test_hotspots = (
                self.get_bird_butterfly_targets(
                    self.df_test, self.config.data.species
                )
            )

        else:
            train_targets = self.get_bird_targets(train_hotspots)
            val_targets = self.get_bird_targets(val_hotspots)
            test_targets = self.get_bird_targets(test_hotspots)
            exclude_train, exclude_val, exclude_test = None, None, None

        self.all_train_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(train_data, dtype=torch.float32),
            targets=torch.tensor(train_targets, dtype=torch.float32),
            exclude=exclude_train,
            hotspots=train_hotspots,
            data_base_dir=self.data_base_dir,
            mode="train",
            maximum_known_labels_ratio=self.config.Ctran.train_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Ctran.quantized_mask_bins,
        )

        self.all_val_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(val_data, dtype=torch.float32),
            targets=torch.tensor(val_targets, dtype=torch.float32),
            exclude=exclude_val,
            hotspots=val_hotspots,
            data_base_dir=self.data_base_dir,
            mode="val",
            maximum_known_labels_ratio=self.config.Ctran.eval_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Ctran.quantized_mask_bins,
        )

        self.all_test_dataset = globals()[self.dataloader_to_use](
            data=torch.tensor(test_data, dtype=torch.float32),
            targets=torch.tensor(test_targets, dtype=torch.float32),
            exclude=exclude_test,
            hotspots=test_hotspots,
            data_base_dir=self.data_base_dir,
            mode="test",
            maximum_known_labels_ratio=self.config.Ctran.eval_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Ctran.quantized_mask_bins,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.all_train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
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

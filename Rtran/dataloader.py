from typing import Any, Callable, Dict, Optional
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as trsfs

from src.dataset.geo import VisionDataset
from src.dataset.utils import load_file, get_subset
from Rtran.label_masking import get_unknown_mask_indices


class SDMMaskedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 target_type="probs", targets_folder="corrected_targets", images_folder="images",
                 env_data_folder="environmental",
                 maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1,
                 quantized_mask_bins=1) -> None:
        """
        this dataloader handles dataset with masks for RTran model, handling a single dataset at a time (SatBird or SatButterly)
        Parameters:
            df: dataframe with hotspot IDs
            data_base_dir: base directory for data
            env: list of env data to take into account [ped, bioclim]
            env_var_sizes: size of environmental variables
            transforms: transforms functions
            mode : train|val|test
            datatype: "refl" (reflectance values ) or "img" (image dataset)
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            images_folder: folder name for sat. images
            env_data_folder: folder name for env data
            maximum_known_labels_ratio: known labels ratio for RTran
            num_species: total number of species/classes to predict
            species_set: set with different species sizes
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """
        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.target_type = target_type
        self.targets_folder = targets_folder
        self.img_folder = images_folder
        self.env_data_folder = env_data_folder
        self.num_species = num_species
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        # loading satellite image
        if self.data_type == 'img':
            img_path = os.path.join(self.data_base_dir, self.img_folder[0] + "_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.img_folder[0], hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        assert len(self.env) == len(self.env_var_sizes), "# of env variables must be equal to the size of env vars "
        # loading environmental rasters, if any
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder[0], hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * self.env_var_sizes[i - 1]
            e_i = self.env_var_sizes[i] + s_i
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image
        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        item_["sat"] = item_["sat"].squeeze(0)
        # constructing targets
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder[0], hotspot_id + '.json'))

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])
        if self.target_type == "binary":
            item_["target"][item_["target"] > 0] = 1

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode,
                                                    max_known=self.maximum_known_labels_ratio,
                                                    predict_family_of_species=self.predict_family_of_species,
                                                    data_base_dir=self.data_base_dir)
        mask = item_["target"].clone()
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)

        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask

        item_["mask_q"] = mask_q
        item_["mask"] = mask
        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        return item_


class SDMCoLocatedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 target_type="probs", targets_folder="corrected_targets", images_folder="images",
                 env_data_folder="environmental",
                 maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1,
                 quantized_mask_bins=1) -> None:
        """
        this dataloader handles dataset SatBird + SatButterfly_v2( co-located with some of SatBird positions)
        Parameters:
            df: dataframe with hotspot IDs
            data_base_dir: base directory for data
            env: list of env data to take into account [ped, bioclim]
            env_var_sizes: size of environmental variables
            transforms: transforms functions
            mode : train|val|test
            datatype: "refl" (reflectance values ) or "img" (image dataset)
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            images_folder: folder name for sat. images
            env_data_folder: folder name for env data
            maximum_known_labels_ratio: known labels ratio for RTran
            num_species: total number of species/classes to predict
            species_set: set with different species sizes
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.target_type = target_type
        self.targets_folder = targets_folder
        self.img_folder = images_folder
        self.env_data_folder = env_data_folder
        self.num_species = num_species
        self.species_set = species_set
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        # loading satellite image
        if self.data_type == 'img':
            img_path = os.path.join(self.data_base_dir, self.img_folder[0] + "_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.img_folder[0], hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        assert len(self.env) == len(self.env_var_sizes), "# of env variables must be equal to the size of env vars "
        # loading environmental rasters, if any
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder[0], hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * self.env_var_sizes[i - 1]
            e_i = self.env_var_sizes[i] + s_i
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image
        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        item_["sat"] = item_["sat"].squeeze(0)
        # constructing targets
        species_2_to_exclude = -1
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder[0], hotspot_id + '.json'))
        if os.path.exists(os.path.join(self.data_base_dir, self.targets_folder[1], hotspot_id + '.json')):
            species_2 = load_file(os.path.join(self.data_base_dir, self.targets_folder[1], hotspot_id + '.json'))
        else:
            species_2 = {}
            species_2["probs"] = [-2] * self.species_set[1]
            species_2_to_exclude = 1

        species["probs"] = species["probs"] + species_2["probs"]

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode,
                                                    max_known=self.maximum_known_labels_ratio,
                                                    absent_species=species_2_to_exclude,
                                                    species_set=self.species_set,
                                                    predict_family_of_species=self.predict_family_of_species,
                                                    data_base_dir=self.data_base_dir)
        mask = item_["target"].clone()
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)

        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask

        mask[mask > 0] = 1
        item_["mask_q"] = mask_q
        item_["mask"] = mask
        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        return item_


class SDMCombinedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 target_type="probs", targets_folder="corrected_targets",
                 targets_folder_2="SatButterfly_dataset/SatButterfly_v2/USA/butterfly_targets_v1.2",
                 images_folder="images", env_data_folder="environmental",
                 maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1,
                 quantized_mask_bins=1) -> None:
        """
        this dataloader handles all datasets together SatBird + SatButterfly_v1+ SatButterfly_v2( co-located with some of SatBird positions)
        Parameters:
            df: dataframe with hotspot IDs
            data_base_dir: base directory for data
            env: list of env data to take into account [ped, bioclim]
            env_var_sizes: size of environmental variables
            transforms: transforms functions
            mode : train|val|test
            datatype: "refl" (reflectance values ) or "img" (image dataset)
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            targets_folder_2: folder name for butterfly targets whenever co-located with ebird
            images_folder: folder name for sat. images
            env_data_folder: folder name for env data
            maximum_known_labels_ratio: known labels ratio for RTran
            num_species: total number of species/classes to predict
            species_set: set with different species sizes
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.target_type = target_type
        self.targets_folder = targets_folder
        self.targets_folder_2 = targets_folder_2
        self.img_folder = images_folder
        self.env_data_folder = env_data_folder
        self.num_species = num_species
        self.species_set = species_set
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']
        folder_index = 0
        # to differentiate between SatBird hotspots, or butterfly-only hotspots (SatButterfly-v1) (starting with "BL")
        if hotspot_id.startswith("BL"):
            folder_index = 1
        # loading satellite image
        if self.data_type == 'img':
            img_path = os.path.join(self.data_base_dir, self.img_folder[folder_index] + "_visual",
                                    hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.img_folder[folder_index], hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        assert len(self.env) == len(self.env_var_sizes), "# of env variables must be equal to the size of env vars "

        # loading environmental rasters, if any
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder[folder_index], hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * self.env_var_sizes[i - 1]
            e_i = self.env_var_sizes[i] + s_i
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image
        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        item_["sat"] = item_["sat"].squeeze(0)
        # constructing targets
        species_to_exclude = -1

        if folder_index == 0:  # SatBird hotspot (which may or may not have butterfly targets)
            species = load_file(
                os.path.join(self.data_base_dir, self.targets_folder[folder_index], hotspot_id + '.json'))

            if os.path.exists(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json')):
                species_2 = load_file(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json'))
            else:
                species_2 = {"probs": [-2] * self.species_set[int(1 - folder_index)]}
                species_to_exclude = 1
        else:  # A butterfly-only hotspot
            species_2 = load_file(
                os.path.join(self.data_base_dir, self.targets_folder[folder_index], hotspot_id + '.json'))
            species = {"probs": [-2] * self.species_set[int(1 - folder_index)]}
            species_to_exclude = 0

        species["probs"] = species["probs"] + species_2["probs"]

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode,
                                                    max_known=self.maximum_known_labels_ratio,
                                                    absent_species=species_to_exclude, species_set=self.species_set,
                                                    predict_family_of_species=self.predict_family_of_species,
                                                    data_base_dir=self.data_base_dir)
        mask = item_["target"].clone()

        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)

        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask

        mask[mask > 0] = 1

        item_["mask_q"] = mask_q
        item_["mask"] = mask
        # meta data
        item_["hotspot_id"] = hotspot_id
        return item_


class sPlotDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = []
        return item_

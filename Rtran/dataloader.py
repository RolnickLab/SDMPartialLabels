import random
from typing import Any, Callable, Dict, Optional
import os
import json

from src.dataset.geo import VisionDataset
from src.dataset.utils import load_file, get_subset

import torch
from torchvision import transforms as trsfs
import numpy as np


def get_unknown_mask_indices(num_labels, mode, max_known=0.5, absent_species=-1,
                             species_set=None, predict_family_of_species=-1,
                             per_species_mask_file="/network/projects/ecosystem-embeddings/SatButterfly_v1/bird_species_family_mapping.json"):
    """
    num_labels: total number of species
    mode: train, val or test
    max_unknown: number of unknown values at max
    absent species: if not -1, birds are absent (0), butterflies are absent (1)
    species set: a list of [bird species, butterfly species]
    predict family of species: (only if not training) mask out either birds or butterflies
    """
    # sample random number of known labels during training; in testing, everything is unknown
    with open(per_species_mask_file, 'r') as f:
        per_species_mask = json.load(f)
        mask_max_size = len(per_species_mask.keys())

    if mode == 'train': # all species are there
        random.seed()
        if absent_species == -1:  # 50% of the time when butterflies are there, mask all butterflies
            if random.random() < 0.5 and species_set is not None:
                absent_species = int(np.random.randint(0, 2, 1)[0]) # 0 or 1
                present_species = 1 - absent_species
                if absent_species == 0:
                    unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                        species_set[absent_species] + (present_species * species_set[present_species]))
                else:
                    unk_mask_indices = np.arange(present_species * species_set[absent_species],
                                        species_set[present_species] + (present_species * species_set[present_species]))
            else:
                # absent_species = int(np.random.randint(0, mask_max_size, 1)[0])
                # unk_mask_indices = np.array(list(per_species_mask.values())[absent_species])
                num_known = random.randint(0, int(num_labels * max_known)) # known: 0, 0.75l -> unknown: l, 0.25l
                unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))
        elif absent_species == 1: # butterflies missing
                present_species = 1 - absent_species

                # what_to_mask = int(np.random.randint(0, mask_max_size, 1)[0])
                # unk_mask_indices = np.array(list(per_species_mask.values())[what_to_mask])
                unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                                species_set[present_species] + (
                                                                            present_species * species_set[
                                                                        absent_species]))),
                                                 int(species_set[present_species] * max_known))
        elif absent_species == 0: #birds unknown
            present_species = 1 - absent_species
            unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                            species_set[absent_species] + (
                                                                        present_species * species_set[
                                                                    present_species]))),
                                             int(species_set[present_species] * max_known))
            # if absent = 1,
            # present = 0
            # max_unknown = 0, 0.5*670
            # index_start = 0
            # index_end = 670 + 0*670
            # if absent = 0, present = 1
            # unknown = 0, 0.5 * 601
            # index_start = 670,
            # index_end = 670 + 1*601

    else:
        # for testing, everything is unknown
        if absent_species == 1:
            present_species = 1 - absent_species
            unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                            species_set[present_species] + (present_species * species_set[absent_species]))),
                                                            int(species_set[present_species] * max_known))
        elif absent_species == 0:
            present_species = 1 - absent_species
            unk_mask_indices = random.sample(list(np.arange(present_species * species_set[absent_species],
                                                            species_set[absent_species] + (present_species * species_set[present_species]))),
                                                            int(species_set[present_species] * max_known))

        else:
            num_known = int(num_labels * max_known)
            unk_mask_indices = random.sample(range(num_labels), int(num_labels - num_known))
        # print(absent_species, int(species_set[present_species] * max_unknown), len(unk_mask_indices), unk_mask_indices)

        if predict_family_of_species != -1:
            # to predict butterflies only
            if predict_family_of_species == 1:
                unk_mask_indices = np.arange(species_set[0], species_set[0] + species_set[1])
            if predict_family_of_species == 0:
                # to predict birds only
                unk_mask_indices = np.arange(0, species_set[0])

    return unk_mask_indices


class SDMMaskedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 targets_folder="corrected_targets", images_folder="images", env_data_folder="environmental",
                 maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1, quantized_mask_bins=1) -> None:
        """
        df_paths: dataframe with hotspot IDs
        data_base_dir: base directory for data
        env: list eof env data to take into account [ped, bioclim]
        transforms: transforms functions
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.targets_folder = targets_folder
        self.img_folder = images_folder
        self.env_data_folder = env_data_folder
        self.num_species = num_species
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
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
        item_["target"][item_["target"] > 0] = 1

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode, max_known=self.maximum_known_labels_ratio)
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
                 targets_folder="corrected_targets", images_folder="images", env_data_folder="environmental",
                maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1, quantized_mask_bins=1) -> None:
        """
        SatBird + SatButterfly co-located with some of ebird positions
        df_paths: dataframe with hotspot IDs
        data_base_dir: base directory for data
        env: list eof env data to take into account [ped, bioclim]
        transforms: transforms functions
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
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
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode, max_known=self.maximum_known_labels_ratio,
                                                    absent_species=species_2_to_exclude, species_set=self.species_set, predict_family_of_species=self.predict_family_of_species)
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


class SDMCombinedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 targets_folder="corrected_targets", targets_folder_2="SatBird_data_v2/USA_summer/butterfly_targets_v1.2", images_folder="images", env_data_folder="environmental",
                 maximum_known_labels_ratio=0.5, num_species=670, species_set=None, predict_family=-1, quantized_mask_bins=1) -> None:
        """
        SatBird + SatButterfly co-located with SatBird + SatButterfly independently from ebird
        df_paths: dataframe with paths (image, image_visual, targets, env_data)to data for each hotspot
        data_base_dir: base directory for data
        env: list eof env data to take into account [ped, bioclim]
        transforms: transforms functions
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
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
        if hotspot_id.startswith("BL"):
            folder_index = 1
        # loading satellite image
        if self.data_type == 'img':
            img_path = os.path.join(self.data_base_dir, self.img_folder[folder_index] + "_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.img_folder[folder_index], hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

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
        # current_species_size = len(species["probs"])

        if folder_index == 0:
            species = load_file(
                os.path.join(self.data_base_dir, self.targets_folder[folder_index], hotspot_id + '.json'))

            if os.path.exists(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json')):
                species_2 = load_file(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json'))
            else:
                species_2 = {"probs": [-2] * self.species_set[int(1 - folder_index)]}
                species_to_exclude = 1
        else:
            species_2 = load_file(
                os.path.join(self.data_base_dir, self.targets_folder[folder_index], hotspot_id + '.json'))
            species = {"probs": [-2] * self.species_set[int(1 - folder_index)]}
            species_to_exclude = 0

        species["probs"] = species["probs"] + species_2["probs"]

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode, max_known=self.maximum_known_labels_ratio,
                                                    absent_species=species_to_exclude, species_set=self.species_set, predict_family_of_species=self.predict_family_of_species)
        mask = item_["target"].clone()

        mask[mask > 0] = 1
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
        item_["hotspot_id"] = hotspot_id
        return item_

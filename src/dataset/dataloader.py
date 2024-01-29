# main data-loader
import os
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch
from torchvision import transforms
import tifffile as tiff

from src.dataset.geo import VisionDataset
from src.dataset.utils import get_subset, load_file, encode_loc
from torchvision import transforms as trsfs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EbirdVisionDataset(VisionDataset):

    def __init__(self,
                 df,
                 data_base_dir,
                 bands,
                 env,
                 env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 mode: Optional[str] = "train",
                 datatype="refl",
                 target="probs",
                 targets_folder="targets",
                 env_data_folder="environmental_bounded",
                 images_folder="images",
                 subset=None,
                 use_loc=False,
                 res=[],
                 loc_type=None,
                 num_species=684,
                 species_set=None,
                 predict_family_of_species=-1) -> None:
        """
        df_paths: dataframe with paths to data for each hotspot
        data_base_dir: base directory for data
        bands: list of bands to include, anysubset of  ["r", "g", "b", "nir"] or  "rgb" (for image dataset) 
        env: list eof env data to take into account [ped, bioclim]
        transforms:
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep 
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.total_images = len(df)
        self.transform = transforms
        self.bands = bands
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.type = datatype
        self.target = target
        self.targets_folder = targets_folder
        self.env_data_folder = env_data_folder
        self.images_folder = images_folder
        self.subset = get_subset(subset, num_species)
        self.use_loc = use_loc
        self.loc_type = loc_type
        self.res = res
        self.num_species = num_species
        self.predict_family_of_species = predict_family_of_species

    def __len__(self):
        return self.total_images

    def _load_spectral_bands(self, hotspot_id):
        # crop the image using above defined transform
        transform = transforms.CenterCrop((64, 64))
        sats = []
        if {"B2", "B3", "B4", "B8"}.issubset(set(self.bands)):
            img_path = os.path.join(self.data_base_dir, self.images_folder, hotspot_id + '_10m.tif')
            img = tiff.imread(img_path)
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1])).astype(float)
            img = transform(torch.from_numpy(img))
            sats.append(img.float())
        if {"B5", "B6", "B7", "B8A", "B11", "B12"}.issubset(set(self.bands)):
            img_path = os.path.join(self.data_base_dir, self.images_folder, hotspot_id + '_20m.tif')
            img = tiff.imread(img_path)
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1])).astype(float)
            img = transform(torch.from_numpy(img))
            sats.append(img.float())
        if {"r", "g", "b"}.issubset(set(self.bands)):
            img_path = os.path.join(self.data_base_dir, self.images_folder, hotspot_id + '_visual.tif')
            img = tiff.imread(img_path)
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1])).astype(float)
            img = transform(torch.from_numpy(img))
            sats.append(img.float())

        sats = torch.cat(sats, dim=0)

        return sats

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        band_groups = []
        assert self.type != "" or len(self.env) == 0, "input cannot be empty of satellite image or env data"
        # satellite image
        if len(set(self.bands)) > 0:
            item_["sat"] = self._load_spectral_bands(hotspot_id=hotspot_id)

        assert len(self.env) == len(self.env_var_sizes), "# of env variables must be equal to the size of env vars "

        # env rasters
        acc_env_var_size = 0
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder, hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * acc_env_var_size
            e_i = self.env_var_sizes[i] + s_i
            acc_env_var_size = self.env_var_sizes[i]
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        t = transforms.Compose(self.transform)
        item_ = t(item_)

        if self.type:
            item_["input"] = item_["sat"]
        else:
            item_["input"] = None

        for e in self.env:
            if item_["input"] is None:
                item_["input"] = item_[e]
            else:
                item_["input"] = torch.cat([item_["input"], item_[e]], dim=-3).float()

        # target labels
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder, hotspot_id + '.json'))
        if self.target == "probs":
            if not self.subset is None:
                item_["target"] = np.array(species["probs"])[self.subset]
            else:
                item_["target"] = species["probs"]
            item_["target"] = torch.Tensor(item_["target"])

        elif self.target == "binary":
            if self.subset is not None:
                targ = np.array(species["probs"])[self.subset]
            else:
                targ = species["probs"]
            item_["original_target"] = torch.Tensor(targ)
            targ[targ > 0] = 1
            item_["target"] = torch.Tensor(targ)

        else:
            raise NameError("type of target not supported, should be probs or binary")

        item_["num_complete_checklists"] = species["num_complete_checklists"]

        item_["hotspot_id"] = hotspot_id

        if self.predict_family_of_species != -1:
            songbird_indices = [
                "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/nonsongbird_indices.npy",
                "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/stats/songbird_indices.npy"]
            unk_mask_indices = np.load(songbird_indices[self.predict_family_of_species])
            target_mask = item_["target"].clone()
            target_mask[target_mask >= 0] = 0
            target_mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=1)
            item_["mask"] = target_mask

        return item_


class SDMCombinedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, bands, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,  mode="train", datatype="refl",
                 target="probs", targets_folder="corrected_targets", targets_folder_2="SatButterfly_v2/USA/butterfly_targets_v1.2", images_folder="images", env_data_folder="environmental",
                 subset=None,  num_species=670, species_set=None, predict_family_of_species=-1) -> None:
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
        self.bands = bands
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.target_type = target
        self.targets_folder = targets_folder
        self.targets_folder_2 = targets_folder_2
        self.img_folder = images_folder
        self.env_data_folder = env_data_folder
        self.num_species = num_species
        self.species_set = species_set
        self.predict_family_of_species = predict_family_of_species
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
        acc_env_var_size = 0
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder[folder_index], hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * acc_env_var_size
            e_i = self.env_var_sizes[i] + s_i
            acc_env_var_size = self.env_var_sizes[i]
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image
        if self.data_type:
            item_["input"] = item_["sat"]
        else:
            item_["input"] = None

        for e in self.env:
            if item_["input"] is None:
                item_["input"] = item_[e]
            else:
                item_["input"] = torch.cat([item_["input"], item_[e]], dim=-3).float()

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
        # print(species["probs"])
        # exit(0)

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])
        mask = item_["target"].clone()
        mask[mask >= 0] = 1
        mask[mask < 0] = 0

        # constructing mask for R-tran
        # if species_to_exclude == 0: # birds missing
        #     unk_mask_indices = np.arange(0, self.species_set[0])
        #     mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=0)
        # elif species_to_exclude == 1: # butterflies missing
        #     unk_mask_indices = np.arange(self.species_set[0], self.species_set[0] + self.species_set[1])
        #     mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=0)

        item_["mask"] = mask
        item_["hotspot_id"] = hotspot_id

        if self.predict_family_of_species != -1:
            target_mask = torch.zeros_like(item_["target"])
            if self.predict_family_of_species == 0: # predict birds
                unk_mask_indices = np.arange(0, self.species_set[0])
            else:
                unk_mask_indices = np.arange(self.species_set[0], self.species_set[0] + self.species_set[1]) # predict butterflies

            target_mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=1)
            item_["mask"] = target_mask

        return item_
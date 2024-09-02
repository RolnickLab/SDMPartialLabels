import abc
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trsfs

from Rtran.label_masking import get_unknown_mask_indices
from Rtran.utils import json_load, load_geotiff, load_geotiff_visual


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



class SDMEnvCombinedDataset(EnvDataset):
    def __init__(
        self,
        df,
        data_base_dir,
        env,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        mode="train",
        target_type="probs",
        targets_folder="corrected_targets",
        maximum_known_labels_ratio=0.5,
        num_species=842,
        species_set=None,
        species_set_eval=None,
        predict_family=-1,
        quantized_mask_bins=1,
    ) -> None:
        """
        this dataloader handles all datasets together SatBird + SatButterfly_v1 + SatButterfly_v2( co-located with some of SatBird positions)
        Parameters:
            df: dataframe with hotspot IDs
            data_base_dir: base directory for data
            env: list of env data to take into account [ped, bioclim]
            transforms: transforms functions
            mode : train|val|test
            target_type : "probs" or "binary"
            targets_folder= [targets_folder_bird,targets_folder_butterfly,targets_folder_butterfly_colocated] 
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
        self.mode = mode
        self.target_type = target_type
        self.targets_folder_bird,self.targets_folder_butterfly,self.targets_folder_butterfly_colocated = targets_folder
       
        self.num_species = num_species
        self.species_set = species_set
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]["hotspot_id"]
        has_bird = self.df.iloc[index]["bird"]
        has_butterfly = self.df.iloc[index]["butterfly"]
        item_ = {}

        hotspot_id = self.df.iloc[index]["hotspot_id"]
        
        item_["env"] = torch.Tensor(list(self.df.iloc[index][self.env]))
      
        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

      
        # constructing targets
        target_bird = [-2] * self.species_set[0]
        target_butterfly = [-2] * self.species_set[1] #len(butterfly species)

        species_to_exclude = -1 #all species present
        
        if has_bird==1:
            targ = json_load(
                os.path.join(
                    self.data_base_dir,
                    self.targets_folder_bird,
                    hotspot_id + ".json",
                ))
            target_bird = targ["probs"]
            
                
            if has_butterfly==1:
                targ = json_load(
                os.path.join(
                    self.data_base_dir,
                    self.targets_folder_butterfly_colocated,
                    hotspot_id + ".json",
                ))
                target_butterfly = targ["probs"]
            
            else: 
                species_to_exclude = 1

        else:
            if has_butterfly==1:
                targ = json_load(
                    os.path.join(
                        self.data_base_dir,
                        self.targets_folder_butterfly,
                        hotspot_id + ".json",
                    ))
                target_butterfly = targ["probs"]
                species_to_exclude = 0
            elif has_butterfly==0:
                raise ValueError("cannot have neither butterflies nor birds targets available")
            

        item_["target"] = target_bird + target_butterfly 
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=self.maximum_known_labels_ratio,
            absent_species=species_to_exclude,
            species_set=self.species_set,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=self.data_base_dir,
        )
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
        item_["eval_mask"] = mask
        # meta data
        item_["hotspot_id"] = hotspot_id
        return item_


class sPlotDataloader(DataLoader):
    def __init__(self, worldclim_filename, soilgrids_filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worldclim_data = pd.read_csv(worldclim_filename)
        self.soilgrids_data = pd.read_csv(soilgrids_filename)

        # Ensuring both datasets have the same length by trimming the longer one
        min_len = min(len(self.worldclim_data), len(self.soilgrids_data))
        self.data1 = self.worldclim_data.head(min_len)
        self.data2 = self.soilgrids_data.head(min_len)

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.soilgrids_data)

    def __getitem__(self, index):
        """Returns the items from both files at the given index."""
        item1 = self.worldclim_data.iloc[index]
        item2 = self.soilgrids_data.iloc[index]
        return item1, item2

    
    
class SDMEnvDataset(EnvDataset):
    def __init__(
        self,
        df,
        data_base_dir,
        env,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        mode="train",
        target_type="probs",
        targets_folder="corrected_targets",
        maximum_known_labels_ratio=0.5,
        species_set=None,
        species_set_eval=None, 
        num_species=670,
        predict_family=-1,
        quantized_mask_bins=1,
    ) -> None:
        """
        this dataloader handles dataset with masks for RTran model using env variables as inpu
        Parameters:
            df: dataframe with hotspot IDs
            data_base_dir: base directory for data
            env: list of env variables names to take into account
            transforms: transforms functions
            mode : train|val|test
            target_type : "probs" or "binary"
            targets_folder: folder name for labels/targets
            maximum_known_labels_ratio: known labels ratio for RTran
            num_species: total number of species/classes to predict
            species_set: sets of species 
            predict_family: -1 for none, 0 if we want to focus on predicting species_set[0], 1 if we want to predict species_set[1]
            quantized_mask_bins: how many bins to quantize the positive (>0) encounter rates
        """
        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.mode = mode
        self.target_type = target_type
        self.targets_folder = targets_folder
        self.num_species = num_species
        self.maximum_known_labels_ratio = maximum_known_labels_ratio
        self.species_set= species_set
        self.species_set_eval = species_set_eval
        self.predict_family_of_species = predict_family
        self.quantized_mask_bins = quantized_mask_bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]["hotspot_id"]
        
        item_["env"] = torch.Tensor(list(self.df.iloc[index][self.env]))
        
        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image

        # constructing targets
        species = json_load(
            os.path.join(
                self.data_base_dir, self.targets_folder[0], hotspot_id + ".json"
            )
        )

        item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])
        if self.target_type == "binary":
            item_["target"][item_["target"] > 0] = 1
       
        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=self.maximum_known_labels_ratio,
            species_set = self.species_set,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=self.data_base_dir,
        )
        
        eval_mask_indices = get_unknown_mask_indices(
            num_labels=self.num_species,
            mode=self.mode,
            max_known=0.0,
            species_set = self.species_set_eval,
            predict_family_of_species=self.predict_family_of_species,
            data_base_dir=self.data_base_dir,
        )
        
        mask = item_["target"].clone()
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)
        
        eval_mask= item_["target"].clone()
        eval_mask.scatter_(dim=0, index=torch.Tensor(eval_mask_indices).long(), value=-1.0)
        
        if self.quantized_mask_bins > 1:
            num_bins = self.quantized_mask_bins
            mask_q = torch.where(mask > 0, torch.ceil(mask * num_bins) / num_bins, mask)
        else:
            mask[mask > 0] = 1
            mask_q = mask
            
            eval_mask[eval_mask > 0] = 1

        item_["mask_q"] = mask_q
        item_["mask"] = mask
        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        
        item_["eval_mask"] = eval_mask
        return item_

    
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
        self.targets_folder = self.config.data.files.targets_folder
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
        self.datatype = self.config.data.datatype

        self.predict_family = self.config.predict_family_of_species
        self.num_species = self.config.data.total_species

        # if we are using either SatBird or SatButterly at a time
        self.dataloader_to_use = self.config.dataloader_to_use

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits"""
        self.all_train_dataset = globals()[self.dataloader_to_use](
            df=self.df_train,
            data_base_dir=self.data_base_dir,
            env=self.env,
            transforms=get_transforms(self.config, "train"),
            mode="train",
            target_type=self.target_type,
            targets_folder=self.targets_folder,
            maximum_known_labels_ratio=self.config.Rtran.train_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Rtran.quantized_mask_bins,
        )

        self.all_val_dataset = globals()[self.dataloader_to_use](
            df=self.df_val,
            data_base_dir=self.data_base_dir,
            env=self.env,
            transforms=get_transforms(self.config, "val"),
            mode="val",
            target_type=self.target_type,
            targets_folder=self.targets_folder,
            maximum_known_labels_ratio=self.config.Rtran.eval_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Rtran.quantized_mask_bins,
            
        )

        self.all_test_dataset = globals()[self.dataloader_to_use](
            df=self.df_test,
            data_base_dir=self.data_base_dir,
            env=self.env,
            transforms=get_transforms(self.config, "val"),
            mode="test",
            target_type=self.target_type,
            targets_folder=self.targets_folder,
            maximum_known_labels_ratio=self.config.Rtran.eval_known_ratio,
            num_species=self.num_species,
            species_set=self.config.data.species,
            species_set_eval=self.config.data.species_eval,
            predict_family=self.predict_family,
            quantized_mask_bins=self.config.Rtran.quantized_mask_bins,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.all_train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.all_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.all_test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
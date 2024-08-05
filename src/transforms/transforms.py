import collections
import numbers
import random
from math import ceil, floor
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import normalize

Module.__module__ = "torch.nn"




def normalize_custom(t, mini=0, maxi=1):
    if len(t.shape) == 3:
        return mini + (maxi - mini) * (t - t.min()) / (t.max() - t.min())

    batch_size = t.shape[0]
    min_t = t.reshape(batch_size, -1).min(1)[0].reshape(batch_size, 1, 1, 1)
    t = t - min_t
    max_t = t.reshape(batch_size, -1).max(1)[0].reshape(batch_size, 1, 1, 1)
    t = t / max_t
    return mini + (maxi - mini) * t


class Normalize:
    def __init__(self, maxchan=True, custom=None, subset=sat, normalize_by_255=False):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        subset: set of inputs on which to apply the normalization (typically env variables and sat would require different normalizations)
        """
        self.maxchan = maxchan
        # TODO make this work with the values of the normalization values computed over the whole dataset
        self.subset = subset

        self.custom = custom
        self.normalize_by_255 = normalize_by_255

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        d = {}
        if self.maxchan:
            for task in self.subset:
                tensor = sample[task]
                sample[task] = tensor / torch.amax(tensor, dim=(-2, -1), keepdims=True)
        # TODO
        if self.normalize_by_255:
            for task in self.subset:
                sample[task] = sample[task] / 255
        else:
            if self.custom:
                means, std = self.custom
                for task in self.subset:
            
                    sample[task] = normalize(sample[task], means, std)
        return sample


class NormalizeEnv:
    def __init__(self, custom=None, subset="env", normalize_by_255=False):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        subset: set of inputs on which to apply the normalization (typically env variables and sat would require different normalizations)
        """
        # TODO make this work with the values of the normalization values computed over the whole dataset
        self.subset = subset
    
        self.custom = custom
        self.normalize_by_255 = normalize_by_255

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        d = {}
        if self.normalize_by_255:
            for task in self.subset:
                sample[task] = sample[task] / 255
        else:
            if self.custom:
                means, std = self.custom
                for task in self.subset:
                    batch_size = sample[task].shape[0]
                    
                    sample[task] = (sample[task] -  Tensor(means)) /Tensor(std)
        return sample




class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size
        """
        self.h, self.w = size

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for s in sample:
            if s in sat:
                sample[s] = F.interpolate(
                    sample[s].float(), size=(self.h, self.w), mode="bilinear"
                )
            elif s in env or s in landuse:

                sample[s] = F.interpolate(
                    sample[s].float(), size=(self.h, self.w), mode="nearest"
                )
        return sample


class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, prob=0.5, max_noise=5e-2, std=1e-2):

        self.max = max_noise
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    noise = torch.normal(0, self.std, sample[s].shape)
                    noise = torch.clamp(sample[s], min=0, max=self.max)
                    sample[s] += noise
        return sample




class GaussianBlurring:
    """Convert the input ndarray image to blurred image by gaussian method.

    Args:
        kernel_size (int): kernel size of gaussian blur method. (default: 3)

    Returns:
        ndarray: the blurred image.
    """

    def __init__(self, prob=0.5, kernel_size=3):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s] = torchvision.transforms.functional.gaussian_blur(
                        sample[s], kernel_size=self.kernel_size
                    )
        return sample


def get_transform(transform_item, mode):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "randomnoise" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomGaussianNoise(
            max_noise=transform_item.max_noise or 5e-2, std=transform_item.std or 1e-2
        )

    elif transform_item.name == "normalize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):

        return Normalize(
            maxchan=transform_item.maxchan,
            custom=transform_item.custom or None,
            subset=transform_item.subset,
            normalize_by_255=transform_item.normalize_by_255,
        )
    elif transform_item.name == "normalize_env" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):

        return NormalizeEnv(
            custom=transform_item.custom or None,
            subset=transform_item.subset,
            normalize_by_255=transform_item.normalize_by_255,
        )

    elif transform_item.name == "resize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):

        return Resize(size=transform_item.size)

    elif transform_item.name == "blur" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return GaussianBlurring(prob=transform_item.p)
 

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    transforms = []

    for t in opts.data.transforms:
  
        transforms.append(get_transform(t, mode))
    transforms = [t for t in transforms if t is not None]
 
    return transforms

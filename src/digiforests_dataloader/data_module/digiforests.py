# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from pathlib import Path
from typing import Callable

from .base_data_module import BaseDataModule
from ..dataset.digiforests import DigiForestsDataset


class DigiForestsDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        dataset_config: dict = {},
        transform: Callable[..., dict] | None = None,
        batch_transform: Callable[..., dict] | None = None,
        pre_transform: Callable[..., dict] | None = None,
        pre_filter: Callable[..., bool] | None = None,
    ):
        """
        Initializes the DigiForestsDataModule for managing data loading and preprocessing.

        Args:
            data_dir (Path): Root directory containing the dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            dataset_config (dict, optional): Configuration parameters for the dataset. Defaults to an empty dict.
            transform (Callable[..., dict], optional): Transformation function applied to training samples. Defaults to None.
            batch_transform (Callable[..., dict], optional): Transformation function applied to batches, usually on GPU. Defaults to None.
            pre_transform (Callable[..., dict], optional): Preprocessing transformations applied before data loading. Defaults to None.
            pre_filter (Callable[..., bool], optional): Filtering function to exclude certain samples. Defaults to None.
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            transform=transform,
            batch_transform=batch_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            prepare_splits=["train", "val"],
            dataset_cls=DigiForestsDataset,
            collate_fn=None,
        )


def mink_collate_fn(batch: list[dict[str, torch.Tensor]]):
    """Custom collation function for MinkowskiEngine compatibility.

    Processes a batch of point cloud dictionaries into a format suitable for
    sparse tensor operations. Specifically handles:
    - Adding batch indices to coordinate tensors
    - Concatenating individual samples into batched tensors

    Args:
        batch: List of dictionaries containing individual sample data.
            Each dict should contain keys:
            'pos', 'intensity', 'semantics', 'instance', 'offset'

    Returns:
        Dictionary with batched tensors containing:
        - pos: (N,4) tensor with (batch_idx, x, y, z) coordinates
        - intensity: (N,1) reflectance values
        - semantics: (N,1) semantic labels
        - instance: (N,1) instance labels
        - offset: (N,3) spatial offsets

    Note: Relies on consistent ordering of dictionary keys in input samples
    """
    batch_dict_values = [data_dict.values() for data_dict in batch]
    # this assumes that the order of keys in the dict is the following
    # i can make it general. but i cba.
    pos_list, intensity_list, semantics_list, instance_list, offset_list = list(
        zip(*batch_dict_values)
    )

    batched_pos_list = []
    for i, pos in enumerate(pos_list):
        batch_idx = i * torch.ones_like(pos[:, 0])
        batched_pos = torch.hstack((batch_idx.reshape(-1, 1), pos))
        batched_pos_list.append(batched_pos)
    batched_pos = torch.cat(batched_pos_list, dim=0)
    batched_intensity = torch.cat(intensity_list, dim=0)
    batched_semantics = torch.cat(semantics_list, dim=0)
    batched_instance = torch.cat(instance_list, dim=0)
    batched_offset = torch.cat(offset_list, dim=0)
    return {
        "pos": batched_pos,
        "intensity": batched_intensity,
        "semantics": batched_semantics,
        "instance": batched_instance,
        "offset": batched_offset,
    }


class MinkowskiDigiForestsDataModule(BaseDataModule):
    """DataModule variant optimized for MinkowskiEngine sparse tensor processing.

    Args (inherited from BaseDataModule):
        data_dir: Path to dataset root directory
        batch_size: Number of samples per batch
        num_workers: Parallel data loading workers
        dataset_config: Dataset configuration parameters
        transform: Per-sample transform applied during loading
        batch_transform: Batch-level transform (on GPU)
        pre_transform: Preprocessing applied before storage
        pre_filter: Sample filtering function
        prepare_splits: Dataset splits to prepare (default: ['train', 'val'])
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        dataset_config: dict = {},
        transform: Callable[..., dict] | None = None,
        batch_transform: Callable[..., dict] | None = None,
        pre_transform: Callable[..., dict] | None = None,
        pre_filter: Callable[..., bool] | None = None,
        prepare_splits: list[str] = ["train", "val"],
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            transform=transform,
            batch_transform=batch_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            prepare_splits=prepare_splits,
            dataset_cls=DigiForestsDataset,
            collate_fn=mink_collate_fn,
        )

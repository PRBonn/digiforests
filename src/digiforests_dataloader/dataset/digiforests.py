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

from enum import Enum
from pathlib import Path

import numpy as np
import torch

from ..utils.cloud import Cloud
from ..utils.logging import logger
from .base_dataset import BaseDataset


class Labels:
    class From(Enum):
        UNLABELLED = 0
        OUTLIER = 1
        GROUND = 2
        TREE = 3
        SHRUB = 4
        AUTO_GROUND = 98
        OTHER = 99

    class SubFrom(Enum):
        STEM = 1
        CANOPY = 2

    class To(Enum):
        IGNORE = 0
        GROUND = 1
        SHRUB = 2
        STEM = 3
        CANOPY = 4

    def __init__(self, semantics: np.ndarray, instance: np.ndarray) -> None:
        self.semantics = semantics
        self.instance = instance

    @classmethod
    def read(cls, label_fp: Path):
        data = np.fromfile(label_fp, dtype=np.uint32)

        # bits 0-7 correspond to semantic class
        semantic_labels = data & 0xFF
        # bits 8-15 correspond to sub semantic labels -> stem canopy
        subsemantic_labels = (data >> 8) & 0xFF
        # remap everything into just semantics
        semantic_labels = cls.remap(semantic_labels, subsemantic_labels)
        # bits 16-31 correspond to instance
        instance_labels = data >> 16
        return cls(semantic_labels, instance_labels)

    @staticmethod
    def remap(sem_labels: np.ndarray, sub_labels: np.ndarray):
        remapped_labels = sem_labels.copy()
        # tree is handled seperately
        from_to = {
            Labels.From.UNLABELLED.value: Labels.To.IGNORE.value,
            Labels.From.OUTLIER.value: Labels.To.IGNORE.value,
            Labels.From.GROUND.value: Labels.To.GROUND.value,
            Labels.From.SHRUB.value: Labels.To.SHRUB.value,
            Labels.From.AUTO_GROUND.value: Labels.To.GROUND.value,
            Labels.From.OTHER.value: Labels.To.IGNORE.value,
        }
        for key, value in from_to.items():
            remapped_labels[remapped_labels == key] = value

        # handling tree
        tree_mask = sem_labels == Labels.From.TREE.value
        stem_mask = np.logical_and(tree_mask, sub_labels == Labels.SubFrom.STEM.value)
        canopy_mask = np.logical_and(
            tree_mask, sub_labels == Labels.SubFrom.CANOPY.value
        )
        remapped_labels[stem_mask] = Labels.To.STEM.value
        remapped_labels[canopy_mask] = Labels.To.CANOPY.value
        # Tree semantic class doesn't exist anymore. so any remaining tree points, should be none, should be ignored
        stem_and_canopy_mask = np.logical_or(stem_mask, canopy_mask)
        tree_but_not_sub_class_mask = np.logical_and(
            tree_mask, np.logical_not(stem_and_canopy_mask)
        )
        if np.any(tree_but_not_sub_class_mask):
            logger.warning(
                "there were remaining unmapped tree values. these are being ignored"
            )
            remapped_labels[tree_but_not_sub_class_mask] = Labels.To.IGNORE.value

        return remapped_labels


class DigiForestsDataset(BaseDataset):
    """Dataset class for handling DigiForests point cloud data.

    Extends BaseDataset with specific processing for DigiForests format.

    Attributes:
        num_classes (int): Number of semantic classes in the dataset

    Args:
        root (Path): Root directory containing raw and processed data
        split (str): Dataset split to use (default: 'trainval')
        mode (str): Processing mode - 'default' or 'pred' (prediction) (default: 'default')
        vds (int | None): Voxel downsampling size, if applicable (default: None)
        include_semantics (bool): Include semantic labels (default: True)
        include_instance (bool): Include instance labels (default: False)
        mock_intensity (bool): Use mock intensity values if true (default: False)
        transform (callable): Transform to apply to processed samples
        pre_transform (callable): Transform to apply during preprocessing
        pre_filter (callable): Filter to apply during preprocessing
        delete_existing_processed_files (bool): Force reprocessing of data (default: False)
        root_to_data_dir_callable (callable): Custom directory resolution function
        return_filename (bool): Include filenames in returned samples (default: False)

    Note:
        The 'pred' mode skips loading of label data for inference scenarios.
    """

    num_classes = len(Labels.To)

    def __init__(
        self,
        root: Path,
        split="trainval",  # defined in the data_split.json
        mode: str = "default",  # can be default or pred
        vds: int | None = None,
        include_semantics: bool = True,
        include_instance: bool = True,
        mock_intensity: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        delete_existing_processed_files: bool = False,
        root_to_data_dir_callable=None,
        return_filename: bool = False,
    ):
        # doktor, reinitialize the state machine
        self.mode = mode
        self.include_semantics = include_semantics
        self.include_instance = include_instance
        self.mock_intensity = mock_intensity

        self.vds = vds

        # rtdd callable needs to be overridable when using pyg dataset wrapper
        # supremely ugly. but i need something working
        root_to_data_dir_callable = (
            root_to_data_dir_callable or self.default_rtdd_callable
        )

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            delete_existing_processed_files=delete_existing_processed_files,
            root_to_data_dir_callable=root_to_data_dir_callable,
            return_filename=return_filename,
        )

    def default_rtdd_callable(self, prefix: str, in_path: Path):
        """Resolve raw or processed data directory, accounting for voxel downsampling.

        Args:
            prefix (str): Either 'raw' or 'processed'
            in_path (Path): Base path to resolve from

        Returns:
            Path: Resolved directory path
        """
        if self.vds is not None:
            return in_path / prefix / f"vds{self.vds}"
        else:
            return in_path / prefix

    def _process_raw_file(self, raw_fp: Path):
        """Process a single raw point cloud file into tensor format.

        Loads point cloud and label data, applies preprocessing, and formats
        for model input based on configured options.

        Args:
            raw_fp (Path): Path to raw point cloud file

        Returns:
            dict: Processed data tensors or None if filtered out

        Raises:
            ValueError: If point counts mismatch between positions and labels

        Note:
            Resulting dict keys depend on initialization options but may include:
            'pos', 'intensity', 'semantics', 'instance'
        """
        cloud = Cloud.load(raw_fp)
        if self.mode != "pred":
            label_fp = ((raw_fp / "../../labels").resolve() / raw_fp.name).with_suffix(
                ".label"
            )
            assert label_fp.exists(), f"{label_fp} does not exist for raw_fp"
            labels = Labels.read(label_fp)

        pos = cloud.points.astype(np.float32)  # astype copies if copy!=False
        # sanity checks on data cant happen here? ex. num of points too low etc.
        pos_t = torch.from_numpy(pos)
        data = {"pos": pos_t}

        # intensity
        if not self.mock_intensity:
            intensity = cloud.get_attribute("intensity").astype(np.float32)
            intensity_t = torch.from_numpy(intensity)
        else:
            intensity_t = torch.ones((pos_t.shape[0], 1)).float()

        data["intensity"] = intensity_t
        # __import__("ipdb").set_trace()

        # data gets modified based on the state machine
        if self.mode != "pred":
            if self.include_semantics:
                # int32 causes issues
                semantics = labels.semantics.astype(np.int64).reshape(-1)
                if semantics.shape[0] != pos.shape[0]:
                    raise ValueError(
                        f"semantics shape {semantics.shape}, pos shape {pos.shape},"
                        " unequal points"
                    )
                semantics_t = torch.from_numpy(semantics)
                data["semantics"] = semantics_t

            if self.include_instance:
                # instance is int64 (long). int32 causes issues
                instance_v = labels.instance.astype(np.int64).reshape(-1)
                instance_v = instance_v - np.min(instance_v)
                if instance_v.shape[0] != pos.shape[0]:
                    raise ValueError(
                        f"instance_v shape {instance_v.shape}, pos shape {pos.shape},"
                        " unequal points"
                    )
                instance_t = torch.from_numpy(instance_v)
                data["instance"] = instance_t

        if self.pre_filter is not None and self.pre_filter(raw_fp, data):
            # pre_filter returns True, means sample should be filtered
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

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

import copy
import torch
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
from functools import cached_property

from ..utils.logging import logger, bar
from ..utils.serialize import TorchSerializedList
from ..utils.io import load_yaml_or_json, write_json


class BaseDataset(Dataset):
    """Base dataset class inspired by PyTorch Geometric's structure.

    Handles raw data processing, splits management, and file path organization.
    Requires specific directory structure:
    - Raw data: {root}/raw
    - Processed data: {root}/processed
    - Split definitions: {root}/raw/data_split.json

    Subclasses must implement `_process_raw_file` for custom data processing.

    Args:
        root (Path): Root directory containing raw/ and processed/ subdirectories
        split (str): Data split to load (must exist in data_split.json). Default: 'train'
        transform (Callable): Per-sample transformation function. Default: None
        pre_transform (Callable): Per-sample preprocessing function. Default: None
        pre_filter (Callable): Sample filtering function. Default: None
        delete_existing_processed_files (bool): Forces reprocessing when True (default). See _ensure_processed_data_existence() for details.
        root_to_data_dir_callable (Callable): Override for custom directory resolution (see default_rtdd_callable() implementation requirements)
        return_filename (bool): Return filenames with samples. Not recommended for training. Default: False

    Attributes:
        num_classes (int): Must be set by subclasses for model validation
        sample_fps (TorchSerializedList): List of processed sample file paths (see get_sample_fps())

    Raises:
        Exception: If requested split not found in data_split.json

    Note:
        data_split.json format: {split_name: [relative_file_paths]}
        Example: {"train": ["scan_1.ply", "scan_2.ply"], "val": ["scan_3.ply"]}
        Key Processing Flow:
        - Raw files processed through _process_raw_file() during initialization
        - Preprocessing controlled by _ensure_processed_data_existence()
    """

    # can be useful to ensure checks that model output channels equal num classes
    num_classes = 0

    def __init__(
        self,
        root: Path,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        delete_existing_processed_files: bool = True,
        root_to_data_dir_callable: Callable[[str, Path], Path] | None = None,
        return_filename: bool = False,
    ):
        """Initializes dataset with specified split and processing configuration."""
        super().__init__()
        # doktor, initialize the state machine
        self.root = Path(root)
        self.split = split
        if split not in self.data_split.keys():
            raise Exception(
                f'split {split} not present in the data split json at {root / "raw" / "data_split.json"}'
            )

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.delete_existing_processed_files = delete_existing_processed_files
        self.root_to_data_dir_callable = root_to_data_dir_callable
        self.return_filename = return_filename
        self._ensure_processed_data_existence()

        self.sample_fps = TorchSerializedList(self.get_sample_fps())

    def default_rtdd_callable(self, prefix: str, in_path: Path):
        """Default implementation for raw/processed directory resolution (requires override).

        Args:
            prefix: Directory type - either 'raw' or 'processed'
            in_path: Root directory path provided during initialization

        Raises:
            NotImplementedError: Must be implemented by subclasses if using custom directory structure
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_fps)

    def __getitem__(self, idx):
        """Loads and transforms a single sample from processed storage.

        Args:
            idx: Sample index in [0, len(dataset))

        Returns:
            Dictionary containing sample data tensors. When return_filename=True,
            includes additional 'filename' key with relative path string.

        Note:
            Applies transform if configured, and optionally adds filename metadata
        """
        sample_fp: Path = self.sample_fps[idx]
        logger.trace(f"loading {sample_fp}")
        data: dict[str, str | torch.Tensor] = torch.load(sample_fp)
        if self.transform is not None:
            data = self.transform(data)
        if self.return_filename:
            data["filename"] = sample_fp.relative_to(self.root).as_posix()
        return data

    @cached_property
    def data_split(self) -> dict[str, list[Path]]:
        """Dataset split configuration loaded from raw/data_split.json.

        Returns:
            Dictionary mapping split names to lists of relative file paths

        Raises:
            FileNotFoundError: If data_split.json is missing from raw directory
        """
        file_path = self.root / "raw" / "data_split.json"
        if not file_path.exists():
            raise Exception(f"File {file_path} is required for a dataset.")
        data_split: dict[str, list[Path]] = load_yaml_or_json(
            self.root / "raw" / "data_split.json"
        )
        return data_split

    @property
    def raw_dir(self):
        """Resolved path to raw data directory.

        Uses root_to_data_dir_callable if provided, otherwise defaults to root/raw
        """
        if self.root_to_data_dir_callable is None:
            return (self.root / "raw").resolve()
        else:
            return self.root_to_data_dir_callable("raw", self.root)

    @property
    def processed_dir(self):
        """Resolved path to processed data directory.

        Uses root_to_data_dir_callable if provided, otherwise defaults to root/processed
        """
        if self.root_to_data_dir_callable is None:
            return (self.root / "processed").resolve()
        else:
            return self.root_to_data_dir_callable("processed", self.root)

    def get_sample_fps(self) -> list[Path]:
        """Validates and returns processed sample paths for current split.

        Returns:
            List of absolute paths to processed sample files

        Raises:
            AssertionError: If any processed files are missing
            FileNotFoundError: If split JSON file is missing
        """
        """gets the files from processed dir as per the state machine"""
        # its actually list[str], but whatever to satisfy mypy
        fp_list: list[Path] = load_yaml_or_json(
            self.processed_dir / f"{self.split}.json"
        )
        # sanity checking
        assert isinstance(fp_list, list), fp_list
        # fp list is the raw file list. need to convert to processed file loc list
        processed_fp_list = copy.deepcopy(fp_list)
        for idx in range(len(fp_list)):
            processed_fp_list[idx] = self.processed_dir / fp_list[idx]
            assert processed_fp_list[
                idx
            ].exists(), f"processed file {processed_fp_list[idx]} doesn't exist"
        return processed_fp_list

    def _ensure_processed_data_existence(self):
        """Ensures processed data exists by converting raw files through preprocessing pipeline.

        Performs the following operations:
        1. Validates existence of all raw files in the current split
        2. Processes raw files through _process_raw_file() implementation
        3. Applies pre_transform and pre_filter if configured
        4. Saves processed data as .pt files in processed directory
        5. Maintains JSON manifest of processed files for the split

        Workflow:
        - Skips existing processed files unless delete_existing_processed_files=True
        - Creates directory structure mirroring raw files in processed dir
        - Generates {split}.json manifest of processed files

        Raises:
            AssertionError: If any raw file is missing or invalid
            FileNotFoundError: If required raw files are not found
        """
        # TODO: somehow dump the dataset configuration as a pickle. and compare that too
        # so that if the config changes, data is reprocessed
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        raw_fp_list = copy.deepcopy(self.data_split[self.split])
        processed_fp_list = []

        for fp_name in raw_fp_list:
            raw_fp: Path = self.raw_dir / fp_name
            assert (
                raw_fp.exists() and raw_fp.is_file()
            ), f"{raw_fp.exists()} and {raw_fp.is_file()}, raw_fp {raw_fp}"
            new_file_name = Path(fp_name).with_suffix(".pt")
            processed_fp = self.processed_dir / new_file_name
            if processed_fp.exists():
                if not self.delete_existing_processed_files:
                    processed_fp_list.append(new_file_name)
                    continue
                else:
                    logger.warning(f"removing existing file at {processed_fp}")
                    processed_fp.unlink()
            data = self._process_raw_file(raw_fp)
            if data is None:
                # prefiltered or to be skipped
                logger.debug("skipping", raw_fp)
                continue
            else:
                processed_fp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(data, processed_fp)
                logger.debug(f"raw file - {raw_fp} - processed to - {processed_fp}")
                processed_fp_list.append(new_file_name)
        write_json(
            processed_fp_list,
            self.processed_dir / f"{self.split}.json",
            overwrite=True,
            verbose=True,
        )

    def _process_raw_file(self, raw_fp: Path) -> dict | None:
        """Abstract method for converting raw file to processed data dict.

        Subclasses must implement this to handle their specific data format.

        Args:
            raw_fp: Absolute path to raw file from raw_dir

        Returns:
            dict: Processed data containing tensors (pos, labels, etc.) or
            None: To skip this sample (filtered out)

        Implementation Requirements:
        1. Load and parse raw file format
        2. Convert to dict of torch.Tensors with standardized keys
        3. Apply pre_transform if configured
        4. Apply pre_filter if configured
        """
        raise NotImplementedError

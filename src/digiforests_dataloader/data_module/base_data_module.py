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

from pathlib import Path
from typing import Callable, Type

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..utils.logging import logger
from ..dataset.base_dataset import BaseDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int,
        dataset_config: dict,
        transform: Callable[..., dict] | None = None,
        batch_transform: Callable[..., dict] | None = None,
        pre_transform: Callable[..., dict] | None = None,
        pre_filter: Callable[..., bool] | None = None,
        dataset_cls: Type[BaseDataset] | None = None,
        prepare_splits: list[str] = ["train", "val", "test", "predict"],
        collate_fn=None,
    ):
        """
        Args:
            data_dir (Path): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of workers for data loading.
            dataset_config (dict): Configuration parameters for the dataset.
            transform (Callable[..., dict], optional): Function to apply transformations to individual samples. Not applied to validation data.
            batch_transform (Callable[..., dict], optional): Function to apply transformations to batches, useful for GPU-based augmentations.
            pre_transform (Callable[..., dict], optional): Function to apply transformations before loading samples.
            pre_filter (Callable[..., bool], optional): Function to filter out unwanted samples.
            dataset_cls (Type[BaseDataset], optional): Custom dataset class to use.
            prepare_splits (list[str]): List of dataset splits to prepare (default: ["train", "val", "test", "predict"]).
            collate_fn (optional): Custom function for collating samples into batches.

        Notes:
            - `transform` is typically applied on the CPU before data transfer to the GPU.
            - `batch_transform` is called in the `on_after_batch_transfer` hook and is useful for GPU-based operations.
            - More details on Lightning data module hooks:
              https://pytorch-lightning.readthedocs.io/en/2.0.1.post0/common/lightning_module.html#hooks
        """
        super().__init__()
        # calling save_hyperparams gives you every init arg as part of the self.hparams dict
        # TODO. im curious how the transforms will be dumped. probably need proper reprs
        # this also necessitates # pyright: ignore all over the place, which is super
        # ugly. need to somehow declare that hparams has those keys and thos types.
        self.save_hyperparameters(
            ignore=[
                "data_dir",
                "transform",
                "batch_transform",
                "pre_transform",
                "pre_filter",
                "dataset_cls",
                "prepare_splits",
                "collate_fn",
            ]
        )
        # store the ignored as attributes
        self.data_dir = data_dir
        self.transform = transform
        self.batch_transform = batch_transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.dataset_cls = dataset_cls if dataset_cls is not None else BaseDataset
        self.prepare_splits = prepare_splits
        self.collate_fn = collate_fn

    def prepare_data(self):
        """
        Download and save data. This method handles one-time processing tasks
        and is always called from the main worker when multiple processes are used.

        Note: Do not assign state here; instead, use `setup` or `__init__`.
        """

        def summarize_dataset(name: str, dataset):
            logger.info("=" * 10)
            logger.info(f"{name} dataset has {len(dataset)} samples")
            logger.info("=" * 10)

        # ensure data state. init-ing below, calls the preprocess stuff
        for split in self.prepare_splits:
            summarize_dataset(
                split,
                self.dataset_cls(
                    root=self.data_dir,
                    split=split,
                    transform=None,
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                    **self.hparams.dataset_config,  # pyright: ignore
                ),
            )

    def setup(self, stage=None):
        # Create datasets and set state
        if stage == "fit":
            self.train_data = self.dataset_cls(
                root=self.data_dir,
                split="train",
                transform=self.transform,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "fit" or stage == "validate":
            self.val_data = self.dataset_cls(
                root=self.data_dir,
                split="val",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "test":
            self.test_data = self.dataset_cls(
                root=self.data_dir,
                split="test",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "predict":
            self.predict_data = self.dataset_cls(
                root=self.data_dir,
                split="predict",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """the batch will have been transferred to device when this is called."""
        if self.trainer.training and self.batch_transform:  # pyright: ignore
            # if self.batch_transform:
            batch = self.batch_transform(batch)
        return batch

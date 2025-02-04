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

import os
import copy
import json
import torch
import typer
from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import digiforests_dataloader.transforms as ddt
from digiforests_dataloader.utils.logging import logger
from digiforests_dataloader.utils.serialize import PathEncoder
from digiforests_dataloader import MinkowskiDigiForestsDataModule
from digiforests_dataloader.utils.io import load_yaml_or_json, write_json, write_yaml

from forest_pan_seg import MinkUNetPanoptic
from forest_pan_seg.utils import ConsoleLogger, sync_config_keys


def train(
    model_config: dict,
    lightning_config: dict,
    datamodule_config: dict,
    log_dir: Path,
    data_dir: Path,
    lightning_loggers: list[Logger],
    ckpt=None,
):
    # Data Prep
    # ----------------------
    # TODO: log all the transforms as well
    def pre_filter(raw_fp, data: dict[str, torch.Tensor]):
        """should return true if the data should be filtered out"""
        unique_iids = data["instance"].unique()
        if len(unique_iids) > 1:
            return False
        else:
            logger.debug(
                "Prefiltering", raw_fp, "because num of instances is", len(unique_iids)
            )
            return True

    pre_transfom = ddt.Compose(
        [ddt.CenterGlobal(verbose=True), ddt.AddOffsets(ignore_id=0)]
    )
    batch_transform = ddt.Compose(
        [
            ddt.RandomRotate(
                180, axis=2, minkowski=True, verbose=True, rotate_vectors=True
            ),
            ddt.RandomReflection(
                axis=2,  # z-axis
                minkowski=True,
                verbose=True,
                reflect_vectors=True,
            ),
            ddt.RandomUniformScale(
                (0.8, 1.2), minkowski=True, verbose=True, scale_vectors=True
            ),
        ]
    )
    dataset_config = datamodule_config.pop("dataset")
    if "debug_no_augmentation" in datamodule_config:
        debug_no_augmentation = datamodule_config.pop("debug_no_augmentation")
        batch_transform = None if debug_no_augmentation else batch_transform

    datamodule = MinkowskiDigiForestsDataModule(
        data_dir=data_dir,
        dataset_config=dataset_config,
        transform=None,
        batch_transform=batch_transform,
        pre_transform=pre_transfom,
        pre_filter=pre_filter,
        **datamodule_config,
    )
    # ----------------------

    # Model
    # ----------------------
    model = MinkUNetPanoptic(**model_config)
    assert (
        model.hparams.num_classes == datamodule.dataset_cls.num_classes
    ), f"model num_classes {model.hparams.num_classes} doesnt match the dataset {datamodule.dataset_cls.num_classes}"

    lr_monitor = LearningRateMonitor(logging_interval="step")
    metric_key: str = lightning_config.pop("checkpoint_metric_key")
    metric_ckpt = ModelCheckpoint(
        dirpath=None,  # use the default_root_dir of trainer
        monitor=metric_key,
        filename=metric_key.replace("/", "_") + "_{epoch:02d}_{" + metric_key + ":.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        verbose=False,
        save_on_train_epoch_end=False,
    )
    # ----------------------

    # Train
    # ----------------------
    # if you have a device that has Tensor Cores
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    del lightning_config["seed"]
    trainer = Trainer(
        logger=lightning_loggers,
        callbacks=[lr_monitor, metric_ckpt],
        default_root_dir=log_dir,
        **lightning_config,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt)
    # ----------------------

    # Test
    # ----------------------
    if "test" in datamodule.prepare_splits:
        trainer.test(model, datamodule=datamodule)


def main(
    model_config_fp: Optional[Path] = typer.Option(
        None,
        "--model-conf",
        help="Path to the model configuration file.",
    ),
    lightning_config_fp: Optional[Path] = typer.Option(
        None,
        "--lightning-conf",
        help="Path to the Lightning configuration file.",
    ),
    data_config_fp: Optional[Path] = typer.Option(
        None,
        "--data-conf",
        help="Path to the data module configuration file.",
    ),
    log_dir: Path = typer.Option(
        ..., help="Root directory for logs: log_dir/experiment_name/run_name."
    ),
    data_dir: Path = typer.Option(
        ..., help="Path to the dataset directory (should contain a ./raw folder)."
    ),
    experiment_name: str = typer.Option(
        ..., help="The name of the experiment, e.g., 'kpconv'."
    ),
    run_name: Optional[str] = typer.Option(
        None, help="The name of a run. Defaults to 'version_#' if not provided."
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode: clean run directory, activate Lightning debug args, and apply other debug settings.",
    ),
    threads: Optional[int] = typer.Option(
        None, help="Number of threads/workers to use for the data module."
    ),
    ckpt: Optional[Path] = typer.Option(
        None, help="Path to a model checkpoint file for resuming training."
    ),
):
    """
    Run training for the DigiForests panoptic segmentation model.

    This function sets up the training environment, loads configurations, and initiates
    the training process for the forest panoptic segmentation model.

    Args:
        model_config_fp: Path to YAML file with model architecture settings.
        lightning_config_fp: Path to YAML file with PyTorch Lightning Trainer settings.
        data_config_fp: Path to YAML file with dataset and dataloader settings.
        log_dir: Root directory for storing experiment logs and checkpoints.
        data_dir: Path to the DigiForests dataset root directory.
        experiment_name: Identifier for the current experiment series.
        run_name: Optional identifier for the specific run (auto-generated if None).
        debug: If True, enables debug mode with additional logging and directory cleaning.
        threads: Override for the number of data loading worker threads.
        ckpt: Path to a checkpoint file for resuming training.

    Workflow:
    1. Set up logging and directory structure
    2. Load and merge configuration files
    3. Apply debug settings if enabled
    4. Ensure configuration consistency across components
    5. Save all configurations for reproducibility
    6. Initialize random seeds for reproducibility
    7. Set up PyTorch Lightning Trainer and loggers
    8. Start the training process

    Output:
    - Trained model checkpoints saved to the run directory
    - TensorBoard logs for monitoring training progress
    - Configuration files saved for reproducibility
    - Console and file logging of the training process

    Note:
    - This script uses Typer for CLI argument parsing
    - Configuration files are expected to be in YAML or JSON format
    - Debug mode affects both the training process and directory management
    """
    # catch the cli args first without modification
    cli_args = copy.deepcopy(locals())
    cli_args["execution_dir"] = os.getcwd()
    log_dir.mkdir(parents=True, exist_ok=True)
    # tb logger handles creation of the run dir, in case no name was passed
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False,
        version=run_name,
    )
    run_dir = Path(tb_logger.log_dir)
    if debug:
        # remove the run dir, since we dont want to bloat the runs
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    cli_args["run_dir"] = run_dir
    logger.add_file_handler(run_dir / "logger.log", level=5)

    # Config
    # ----------------------
    model_config = (
        load_yaml_or_json(model_config_fp) if model_config_fp is not None else {}
    )
    datamodule_config = (
        load_yaml_or_json(data_config_fp) if data_config_fp is not None else {}
    )
    lightning_config: dict = (
        load_yaml_or_json(lightning_config_fp)
        if lightning_config_fp is not None
        else {}
    )

    # process and overwrite confs
    if "debug" in lightning_config:
        # debug sub dicts of config should be put at top level
        if debug:
            lightning_config.update(lightning_config["debug"])
        del lightning_config["debug"]
    if "debug" in model_config:
        if debug and model_config["debug"] is not None:
            model_config.update(model_config["debug"])
        del model_config["debug"]
    if threads is not None:
        if (
            "num_workers" in datamodule_config
            and threads != datamodule_config["num_workers"]
        ):
            logger.warning(
                f"Datamodule config num_workers = {datamodule_config['num_workers']}, overwriting with cli arg {threads}"
            )
        datamodule_config["num_workers"] = threads
    # ensure good configuration
    sync_config_keys(
        [cli_args, lightning_config, model_config, datamodule_config], ["batch_size"]
    )
    # in case certain keys have not been defined which we need, define them with defaults
    if "seed" not in lightning_config:
        lightning_config["seed"] = 42
    if "checkpoint_metric_key" not in lightning_config:
        lightning_config["checkpoint_metric_key"] = "val/Mean_IOU"
    if "dataset" not in datamodule_config:
        datamodule_config["dataset"] = {}

    # dump configs
    # will be redundant with lightning's dump, but better safe than sorry
    logger.debug(f"cli_args {json.dumps(cli_args, indent=4, cls=PathEncoder)}")
    write_json(cli_args, run_dir / "cli_config.json")
    logger.debug(
        f"lightning config {json.dumps(lightning_config, indent=4, cls=PathEncoder)}"
    )
    write_yaml(lightning_config, run_dir / "lightning_config.yaml")
    logger.debug(f"model config {json.dumps(model_config, indent=4, cls=PathEncoder)}")
    write_yaml(model_config, run_dir / "model_config.yaml")
    logger.debug(
        f"datamodule config {json.dumps(datamodule_config, indent=4, cls=PathEncoder)}"
    )
    write_yaml(datamodule_config, run_dir / "datamodule_config.yaml")

    # ----------------------
    # setup stuff handled, now on to actual training

    seed_everything(lightning_config["seed"], workers=True)
    data_dir = data_dir.resolve()
    logger.debug("----------------------")
    logger.debug("Training...")
    train(
        model_config=model_config,
        lightning_config=lightning_config,
        datamodule_config=datamodule_config,
        log_dir=run_dir,
        data_dir=data_dir,
        lightning_loggers=[tb_logger, ConsoleLogger()],
        ckpt=ckpt,
    )


if __name__ == "__main__":
    typer.run(main)

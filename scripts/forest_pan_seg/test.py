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
import json
import typer
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything

import digiforests_dataloader.transforms as ddt
from digiforests_dataloader import DigiForestsDataset
from digiforests_dataloader.utils.logging import logger
from digiforests_dataloader.data_module.digiforests import mink_collate_fn

from forest_pan_seg import MinkUNetPanoptic

app = typer.Typer(rich_markup_mode="markdown")


def prepare_dataloader(data_dir):
    pre_transform = ddt.Compose(
        [ddt.CenterGlobal(verbose=True), ddt.AddOffsets(ignore_id=0)]
    )

    dataset = DigiForestsDataset(
        root=data_dir,
        mode="scan",
        split="test",
        transform=None,
        pre_transform=pre_transform,
        include_semantics=True,
        include_instance=True,
        delete_existing_processed_files=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        collate_fn=mink_collate_fn,
    )

    return dataloader


def prepare_model(ckpt_path: Path):
    assert ckpt_path.exists(), f"{ckpt_path} doesn't exist"
    # load the model
    model = MinkUNetPanoptic.load_from_checkpoint(ckpt_path)
    model.to("cuda")
    return model


def transfer_batch_to_device(batch, device="cuda"):
    device_batch = {}
    for key, value in batch.items():
        device_batch[key] = value.to(device)
    return device_batch


def test(data_dir: Path, ckpt_path: Path):
    model = prepare_model(ckpt_path=ckpt_path)
    inference_dataloader = prepare_dataloader(data_dir=data_dir)
    assert (
        model.hparams.num_classes == inference_dataloader.dataset.num_classes
    ), f"model num_classes {model.hparams.num_classes} doesnt match the dataset {inference_dataloader.dataset.num_classes}"
    for idx, batch in enumerate(tqdm(inference_dataloader)):
        batch = transfer_batch_to_device(batch)
        assert (
            batch["pos"][:, 0].unique().shape[0] == 1
        ), "only one sample in batch please."
        model.test_step(batch, idx)
    test_metrics = {}
    test_metrics.update(model.test_seg_metrics.compute())
    test_metrics.update(model.test_pan_metrics.compute())
    # we dont need some stuff
    del (
        test_metrics["test/panopticquality_Ignore"],
        test_metrics["test/panopticquality_Ground"],
        test_metrics["test/panopticquality_Shrub"],
    )
    # we need the mean PQ
    test_metrics["Mean_PQ"] = (
        test_metrics["test/multiclassjaccardindex_Ground"]
        + test_metrics["test/multiclassjaccardindex_Shrub"]
        + test_metrics["test/panopticquality_Tree"]
    ) / 3
    print(
        json.dumps({key: value.item() for key, value in test_metrics.items()}, indent=4)
    )
    return test_metrics


@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    data_dir: Path = typer.Option(
        ..., help="Path to the dataset directory (should contain a ./raw folder)."
    ),
    run_dir: Path = typer.Option(
        None,
        help="Path to the run directory containing hparams.yml. Can alternatively pass ckpt_path.",
    ),
    ckpt_path: Path = typer.Option(None, help="Path to a specific checkpoint file."),
):
    """
    Run testing for the DigiForests panoptic segmentation model.

    This function loads a trained model checkpoint and evaluates its performance
    on a specified test dataset.

    \n\n**Args:**\n
    - `seed`: Random seed for ensuring reproducibility across runs.\n
    - `data_dir`: Path to the root directory of the DigiForests dataset.\n
    - `run_dir`: Directory containing training run artifacts, including checkpoints.
                 Used if ckpt_path is not provided.\n
    - `ckpt_path`: Direct path to a specific model checkpoint file. Takes precedence over run_dir.

    \n\n**Workflow:**\n
    1. Set global random seed for reproducibility\n
    2. Resolve and validate input paths\n
    3. If ckpt_path is not provided, select the latest checkpoint from run_dir\n
    4. Load the trained model from the checkpoint\n
    5. Prepare the test dataloader\n
    6. Run inference on the test dataset\n
    7. Compute and display test metrics

    \n\n**Output:**\n
    - Prints a JSON-formatted dictionary of test metrics including:\n
      - Semantic segmentation metrics (e.g., Mean IOU)\n
      - Panoptic quality metrics for relevant classes\n
      - Mean Panoptic Quality (PQ)

    \n\n**Note:**\n
    - Either run_dir or ckpt_path must be provided\n
    - The function assumes a specific dataset structure and model compatibility

    \n\n**Example usage:**\n
    python test.py --data-dir /path/to/digiforests --run-dir /path/to/training/run
    """

    seed_everything(seed, workers=True)

    data_dir = data_dir.resolve()
    cli_args = copy.deepcopy(locals())
    logger.debug(f"cli_args: {cli_args}")

    if ckpt_path is None:
        assert run_dir is not None, "one of ckpt_path or run_dir is needed"
        ckpts = (run_dir / "checkpoints").glob("*_*.ckpt")
        sorted_ckpts = sorted(ckpts, key=lambda path: path.stem.split("_")[-1])
        ckpt_path = sorted_ckpts[-1]
        logger.info("using", ckpt_path)

    logger.debug("---")
    logger.debug("Testing...")

    test(data_dir=data_dir, ckpt_path=ckpt_path)


if __name__ == "__main__":
    app()

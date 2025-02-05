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
import typer
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything

import digiforests_dataloader.transforms as ddt
from digiforests_dataloader import DigiForestsDataset
from digiforests_dataloader.utils.logging import logger

from forest_pan_seg import MinkUNetPanoptic


app = typer.Typer(rich_markup_mode="markdown")


def collate_fn(batch: list[dict[str, Tensor]]):
    """
    by default, the digiforests dataloader returns a dict of key value tensors.
    the batch here will be a list of such dicts, with an additional filename key.
    the forest pan seg model expects a single dict with original keys and batched tensors.
    """
    filenames = [sample.pop("filename") for sample in batch]
    batch_dict_values = [data_dict.values() for data_dict in batch]
    # this assumes that the order of keys in the dict is the following
    pos_list, intensity_list = list(zip(*batch_dict_values))

    batched_pos_list = []
    for i, pos in enumerate(pos_list):
        batch_idx = i * torch.ones_like(pos[:, 0])
        batched_pos = torch.hstack((batch_idx.reshape(-1, 1), pos))
        batched_pos_list.append(batched_pos)
    batched_pos = torch.cat(batched_pos_list, dim=0)
    batched_intensity = torch.cat(intensity_list, dim=0)

    return {
        "filename": filenames,
        "pos": batched_pos,
        "intensity": batched_intensity,
    }


def prepare_dataloader(data_dir):
    pre_transform = ddt.Compose([ddt.CenterGlobal(verbose=True)])

    dataset = DigiForestsDataset(
        root=data_dir,
        mode="pred",
        split="pred",
        transform=None,
        pre_transform=pre_transform,
        return_filename=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return dataloader


def prepare_model(ckpt_path: Path):
    assert ckpt_path.exists(), f"{ckpt_path} doesn't exist"
    # load the model
    model = MinkUNetPanoptic.load_from_checkpoint(ckpt_path)
    model.to("cuda")
    model.train(False)
    return model


def transfer_batch_to_device(batch, device="cuda"):
    device_batch = {}
    for key, value in batch.items():
        device_batch[key] = value.to(device)
    return device_batch


def predict(data_dir: Path, ckpt_path: Path):
    inference_dataloader = prepare_dataloader(data_dir=data_dir)
    model = prepare_model(ckpt_path=ckpt_path)
    assert (
        model.hparams.num_classes == inference_dataloader.dataset.num_classes
    ), f"model num_classes {model.hparams.num_classes} doesnt match the dataset {inference_dataloader.dataset.num_classes}"

    file_to_preds = {}
    for idx, batch in enumerate(tqdm(inference_dataloader)):
        filenames = batch.pop("filename")
        batch = transfer_batch_to_device(batch)
        preds = model.predict_step(batch, idx)
        file_to_preds[filenames[0]] = preds
    return file_to_preds


def get_sem_conf(pred_sem_conf: np.ndarray):
    # input is probabilities, we use bits 8-15 for conf scores
    quantized_conf = pred_sem_conf * 255
    quantized_conf = np.round(quantized_conf).astype(np.uint32)
    return quantized_conf


def np_arrays_to_binary(semantics, instance, sem_conf=None):
    if sem_conf is not None:
        # bits 0 - 7 are semantics, 8 - 15 are sem conf scores, 16 - 31 are instance
        binary_data = ((instance << 16) | (sem_conf << 8) | semantics).astype(np.uint32)
    else:
        # bits 0 - 15 are semantics, 16 - 31 are instance
        binary_data = ((instance << 16) | semantics).astype(np.uint32)

    return binary_data


@app.command()
def main(
    include_sem_conf: bool = typer.Option(
        True,
        "--conf",
        help="Include semantic confidence scores in the dumped binary format labels.",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    data_dir: Path = typer.Option(
        ..., help="Path to the dataset directory (should contain a ./raw folder)."
    ),
    run_dir: Path = typer.Option(
        None,
        help="Path to the run directory containing hparams.yml. Can alternatively provide ckpt_path.",
    ),
    ckpt_path: Path = typer.Option(
        None,
        help="Path to a specific checkpoint file. If not provided, inferred from run_dir.",
    ),
):
    """
    Run inference on the DigiForests dataset using a trained panoptic segmentation model.

    This function loads a trained model, performs inference on a specified dataset,
    and saves the predictions in a binary format.

    \n\n**Args:**\n
    - `include_sem_conf`: If True, includes semantic confidence scores in the output.\n
    - `seed`: Random seed for ensuring reproducibility.\n
    - `data_dir`: Path to the root directory of the DigiForests dataset.\n
    - `run_dir`: Directory containing training run artifacts, including checkpoints.
                 Used if ckpt_path is not provided.\n
    - `ckpt_path`: Direct path to a specific model checkpoint file. Takes precedence over run_dir.

    \n\n**Workflow:**\n
    1. Set global random seed for reproducibility\n
    2. Resolve and validate input paths\n
    3. If ckpt_path is not provided, select the latest checkpoint from run_dir\n
    4. Load the trained model from the checkpoint\n
    5. Prepare the inference dataloader\n
    6. Run inference on the dataset\n
    7. Process and save predictions in binary format

    \n\n**Output:**\n
    - Saves binary label files (.label) in the dataset structure:
      data_dir/raw/{plot}/inference_labels/{scan}.label\n
    - Binary format:\n
      - If include_sem_conf is True:
        bits 0-7: semantics, 8-15: semantic confidence, 16-31: instance\n
      - If include_sem_conf is False:
        bits 0-15: semantics, 16-31: instance

    \n\n**Note:**\n
    - Either run_dir or ckpt_path must be provided\n
    - The function assumes a specific dataset structure and model compatibility\n
    - Predictions are saved in the original dataset structure for easy comparison

    \n\n**Example usage:**\n
    python inference.py --data-dir /path/to/digiforests --run-dir /path/to/training/run --conf
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
    logger.debug("using", ckpt_path)
    logger.debug("---")
    logger.debug("Predicting...")

    with torch.no_grad():
        labels = predict(data_dir=data_dir, ckpt_path=ckpt_path)

    for filename, label_set in labels.items():
        fp = Path(filename)
        new_fp = Path("raw", *fp.parts[1:-2])
        label_folder = data_dir / new_fp / "inference_labels"
        label_folder.mkdir(parents=True, exist_ok=True)
        label_filename = (label_folder / fp.name).with_suffix(".label")

        pred_sem: np.ndarray = label_set["seg_sem"].cpu().numpy()
        pred_sem_conf: np.ndarray = label_set["seg_sem_conf"].cpu().numpy()
        discrete_pred_sem_conf = get_sem_conf(pred_sem_conf)
        pred_inst: np.ndarray = label_set["pq_inst"].cpu().numpy()

        if include_sem_conf:
            binary_data = np_arrays_to_binary(
                pred_sem, pred_inst, discrete_pred_sem_conf
            )
        else:
            binary_data = np_arrays_to_binary(pred_sem, pred_inst)

        binary_data.astype(np.uint32).tofile(label_filename)
        logger.debug("wrote", label_filename)


if __name__ == "__main__":
    app()

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
import typer
from pathlib import Path

from digiforests_dataloader.utils.io import write_json
from digiforests_dataloader.utils.logging import logger

app = typer.Typer(rich_markup_mode="markdown")


@app.command()
def split(
    raw_folder: Path = typer.Argument(
        ..., help="Path to the raw data folder containing DigiForests dataset."
    ),
    output_fp: Path | None = typer.Option(
        None, help="Optional path to save the data split JSON file."
    ),
):
    """
    Split the DigiForests dataset into train, validation, test, and prediction sets.

    This function organizes the DigiForests dataset into predefined splits based on
    experiment folders. It generates a JSON file containing file paths and statistics
    for each split.

    \n\n**Args:**\n
    - `raw_folder`: Root directory of the DigiForests dataset containing experiment folders.\n
    - `output_fp`: Custom path to save the output JSON file. If None, saves to raw_folder/data_split.json.

    \n\n**Splits:**\n
    - Train: Primary training data from multiple seasons.\n
    - Validation: Held-out data for model tuning.\n
    - Test: Unseen data for final model evaluation.\n
    - Prediction: Specific subset (Spring 2023) for inference tasks.

    \n\n**Output JSON Structure:**\n
    - File counts for each split\n
    - Train/Val ratios\n
    - File paths for each split\n
    - Combined trainval set

    \n\n**Note:**\n
    - Split ratios are calculated based on train and validation sets only.\n
    - The function assumes a specific folder structure within the raw_folder.
    """

    train_exp_folders = [
        "2023-03/exp06-m3",
        "2023-03/exp07-m1",
        "2023-03/exp09-m5",
        "2023-03/exp20-d2",
        "2023-10/exp20-d2",
        "2024-07/exp20-d2",
    ]

    val_exp_folders = [
        "2023-03/exp11-c1",
        "2023-10/exp11-c1",
        "2024-07/exp11-c1",
    ]

    test_exp_folders = [
        "2023-03/exp04-m2",
        "2023-10/exp04-m2",
        "2024-07/exp04-m2",
    ]
    # 2023-03 folders only
    pred_exp_folders = [
        "2023-03/exp04-m2",
        "2023-03/exp06-m3",
        "2023-03/exp07-m1",
        "2023-03/exp09-m5",
        "2023-03/exp11-c1",
        "2023-03/exp20-d2",
    ]
    exp_folders = [
        train_exp_folders,
        val_exp_folders,
        test_exp_folders,
        pred_exp_folders,
    ]

    train_files = []
    val_files = []
    test_files = []
    pred_files = []
    exp_files = [train_files, val_files, test_files, pred_files]

    trainval_files = []

    for idx, exp_folder_list in enumerate(exp_folders):
        for folder in exp_folder_list:
            files = list((raw_folder / folder / "individual_clouds").glob("*.pcd"))
            relative_files = [file_path.relative_to(raw_folder) for file_path in files]
            exp_files[idx].extend(relative_files)

    num_train, num_val, num_test, num_pred = map(len, exp_files)

    trainval_files.extend(train_files)
    trainval_files.extend(val_files)
    num_trainval = len(trainval_files)

    data_dict = {
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "num_pred": num_pred,
        "train_ratio": len(train_files) / num_trainval,
        "val_ratio": len(val_files) / num_trainval,
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "pred": pred_files,
        "trainval": trainval_files,
    }
    json_fp = output_fp or raw_folder / "data_split.json"
    write_json(data_dict, json_fp, sort_keys=False, overwrite=True)

    logger.info(
        f"train ratio: {data_dict['train_ratio']}, val_ratio: {data_dict['val_ratio']}, trainval files: {num_trainval}"
    )
    logger.info(
        "num_train",
        num_train,
        "num_val",
        num_val,
        "num_test",
        num_test,
        "num_pred",
        num_pred,
    )
    logger.info(f"{json_fp} written with the data split")


if __name__ == "__main__":
    app()

# Using DigiForests and its Dataloader

## Overview

We provide a comprehensive toolkit for the DigiForests point cloud dataset, including a dataloader for machine learning research and preprocessing tools for data manipulation.

## Installation

```bash
# Clone the repository
git clone https://github.com/PRBonn/digiforests.git

# Install the package
pip install .
```

## Basic Usage

```python
from digiforests_dataloader import DigiForestsDataModule

# Initialize datamodule. Note that the data_dir should contain a raw/data_split.json
datamodule = DigiForestsDataModule(
    data_dir="/path/to/dataset",
    split="train"
)

# Use in PyTorch Lightning Trainer
trainer.fit(model, datamodule=datamodule)
```

See `data/dataloader.py` for additional details.

**Important Note:** To use our dataloader, please ensure that the dataset corresponds to the [Dataset Configuration](#dataset-configuration) given below. Especially, the `raw` folder needs to contain a `data_split.json`. You can either use the one provided with this repository in `<digiforests_repository>/data/data_split.json` or you can generate one following the instructions in the [Split Configuration](#split-configuration) section.


## Dataset Configuration

The dataset requires a specific folder structure:

```
dataset/
├── raw/
│   ├── data_split.json
│   └── point_clouds/
└── processed/
```

### Custom Splits

To create a custom split of the dataset, use the provided script:

```bash
python scripts/data/split_dataset.py /path/to/dataset/raw [--output-fp /path/to/output.json]
```

This script splits the DigiForests dataset into train, validation, test, and prediction sets.

### Split Configuration

To modify the dataset splits, edit the `split` function in `scripts/data/split_dataset.py`:

```python
def split(...):
    train_exp_folders = [
        "2023-03/exp06-m3",
        "2023-03/exp07-m1",
        # Add more folders for training
    ]

    val_exp_folders = [
        "2023-03/exp11-c1",
        "2023-10/exp11-c1",
        # Add more folders for validation
    ]

    # Similarly, modify test_exp_folders and pred_exp_folders
```

After modifying the split configuration, run the script again to generate the updated `data_split.json`.

### Point Cloud Aggregation

To aggregate individual point clouds and labels:

```bash
python scripts/data/aggregate_clouds_and_labels.py /path/to/plot/folder /path/to/output/folder [--denoise] [--voxel-down-sample-size FLOAT]
```

## Advanced Usage

### Minkowski Engine Support

```python
from digiforests_dataloader import MinkowskiDigiForestsDataModule

datamodule = MinkowskiDigiForestsDataModule(
    data_dir="/path/to/dataset"
)
```

## Performance Tips

- Use `num_workers` for parallel data loading
- Consider using GPU for augmentations (see `batch_transform`)

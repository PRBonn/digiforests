# Forest Panoptic Segmentation Training Scripts

This directory contains scripts for training, testing, and inference of panoptic segmentation models (`forest_pan_seg`) on the [DigiForests Dataset](https://www.ipb.uni-bonn.de/data/digiforest-dataset/).

## Overview

Scripts included:

- `train.py`: Main training script with PyTorch Lightning integration
- `test.py`: Model evaluation on test sets
- `inference.py`: Inference pipeline with prediction export

The DigiForests dataset contains:

- Longitudinal LiDAR scans from 3 seasons (Spring '23, Autumn '23, Summer '24)
- Semantic annotations (Ground/Shrub/Tree)
- Instance-level tree annotations
- Fine-grained stem/canopy labels
- Reference forestry measurements (DBH, crown volume, etc.)

See `scripts/dbh_estimation/README.md` for more on how to estimate tree DBH on the DigiForests data.

A pre-trained model is available [here](http://ipb.uni-bonn.de/html/deeplearningmodels/malladi2025icra/forest_pan_seg_model.ckpt).

## Installation

From the package root, run the following if you haven't already

```
pip install .
```

## Training

```
python scripts/forest_pan_seg/train.py \
  --data-dir /path/to/digiforests_dataset \
  --experiment-name panoptic_v1 \
  --log-dir ./logs \
  [--model-conf configs/model.yaml] \
  [--lightning-conf configs/lightning.yaml] \
  [--data-conf configs/data.yaml]
```

**Key Options**:

- `--model-conf`: Optional Model architecture override config (YAML)
- `--lightning-conf`: Optional Trainer parameters override config (YAML)
- `--data-conf`: Data module settings override config (YAML)
- `--debug`: Enable debug mode (clean logs, etc.)
- `--ckpt`: Resume from the provided checkpoint

**Example Configs**:

```
# model.yaml
in_channels: 1 # intensity as feature
num_classes: 5 # include ignore in the number
coord_dimension: 3
lr: 0.001
batch_size: 1
vds: 0.1
clear_torch_cache: true
cluster_voxel_coords: false
pq_metrics_every: 6

# data.yaml
batch_size: 1
num_workers: 1

# lightning.yaml
seed: 42
max_epochs: 1199
checkpoint_metric_key: val/Mean_IOU
num_sanity_val_steps: 2
devices: 1
accelerator: gpu
precision: 32
log_every_n_steps: 1
gradient_clip_val: 0.5
accumulate_grad_batches: 32
deterministic: false
check_val_every_n_epoch: 2
```

## Testing

```
python scripts/forest_pan_seg/test.py \
  --data-dir /path/to/digiforests_dataset \
  --run-dir ./logs/panoptic_v1/version_0 \
  [--ckpt-path ./best.ckpt]
```

**Metrics Tracked**:

- Segmentation IOU
- Panoptic Quality (PQ)

## Inference

```
python scripts/forest_pan_seg/inference.py \
  --data-dir /path/to/new_forest_scans \
  --run-dir ./logs/panoptic_v1/version_0 \
  [--ckpt-path ./best.ckpt] \
  [--conf]  # Include confidence scores
```

**Outputs**:

- `.label` files with panoptic predictions
- Semantic segmentation confidence scores (optional\*)
- Visualizations in PLY format

\* Note that to run the `dbh_estimation` approach on the output of forest_pan_seg, the confidence scores computed during inference are required (pass the `--conf` flag).

A pre-trained model is provided [here](http://ipb.uni-bonn.de/html/deeplearningmodels/malladi2025icra/forest_pan_seg_model.ckpt) which you can place in the location `<digiforests_repository>/models/forest_pan_seg_model.ckpt`. It can be used with both the testing and inference scripts.

Every script has it's own documentation available by using the `--help` flag.

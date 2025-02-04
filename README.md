<div align="center">
  <h1>DigiForests Dataset Development Kit</h1>
    <a href="#setup"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="#usage"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://www.ipb.uni-bonn.de/pdfs/malladi2025icra.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
</div>

![Teaser](https://www.ipb.uni-bonn.de/data/digiforest-dataset/devkit_teaser.jpg)

The DigiForests dataset provides LiDAR point clouds collected with a backpack-carried mobile mapping system and aerial scanning system. It includes semantic annotations for trees, shrubs, and ground, as well as tree instance annotations and fine-grained semantics for tree stems and crowns.

This development kit offers utilities for handling the DigiForests dataset and includes tools for training panoptic segmentation models and estimating tree DBH (Diameter at Breast Height).

## Project Structure

```
digiforests
├── docker/               # Docker configuration files
├── models/               # Pre-trained model checkpoints
├── scripts/              # Utility scripts for data processing and model evaluation
│   ├── data/
│   ├── dbh_estimation/
│   └── forest_pan_seg/
├── src/                  # Source code for the development kit
│   ├── digiforests_dataloader/
│   ├── forest_pan_seg/
│   └── tree_dbh_estimation/
├── tests/                # Unit tests
```

## Setup

1. [Download](https://www.ipb.uni-bonn.de/data/digiforest-dataset/) the DigiForests dataset.
2. Clone this repository.
3. Install the package:
   ```
   pip install -e .
   ```
4. Explore the data and tools provided in the `scripts/` directory.

## Features

- **Data Loading**: Efficient data loading utilities for the DigiForests dataset.
- **Panoptic Segmentation**: Tools for training and evaluating panoptic segmentation models.
- **DBH Estimation**: Scripts for estimating tree diameter at breast height.
- **Docker Support**: Containerized environment for reproducible research.

## Usage

Refer to the README files in each script directory for specific usage instructions:

- [Data Processing Scripts](scripts/data/README.md)
- [DBH Estimation](scripts/dbh_estimation/README.md)
- [Forest Panoptic Segmentation](scripts/forest_pan_seg/README.md)

## Docker

For a containerized environment, see the [Docker README](docker/README.md) for setup and usage instructions.

## License

This project is free software made available under the MIT license. For details, see the [LICENSE](LICENSE) file.

## Citation

If you use this dataset or development kit in your research, please cite:

````bibtex
@inproceedings{malladi2025icra,
author = {M.V.R. Malladi and N. Chebrolu and I. Scacchetti and L. Lobefaro and T. Guadagnino and B. Casseau and H. Oh and L. Freissmuth and M. Karppinen and J. Schweier and S. Leutenegger and J. Behley and C. Stachniss and M. Fallon},
title = {{DigiForests: A Longitudinal LiDAR Dataset for Forestry Robotics}},
booktitle = {Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA)},
year = {2025},
note = {Accepted},
}```
````

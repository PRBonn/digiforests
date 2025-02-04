# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss
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
from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.logging import logger

from .fit_cylinders import fit_cylinders


def estimate_dbh(cloud: Cloud, voxel_size: float, ransac_iterations: int = 10000):
    """returns a list of tuples and a list of cylinders. in the first, each is a tuple of an xyz ndarray and dbh"""
    pred_cylinders = fit_cylinders(
        cloud, ransac_iterations=ransac_iterations, voxel_size=voxel_size
    )
    pred_xyz_array = [cyl.center.ravel() for cyl in pred_cylinders]
    pred_dbh_array = [2 * cyl.radius for cyl in pred_cylinders]
    xyz_dbh_list = list(zip(pred_xyz_array, pred_dbh_array))
    return xyz_dbh_list, pred_cylinders


def write_dbh_csv(xyz_dbh_list: list[tuple], output_fp: Path):
    import csv

    with open(output_fp.as_posix(), "w", newline="") as csvfile:
        fieldnames = ["id", "x", "y", "z", "dbh"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for idx, (xyz, dbh) in enumerate(xyz_dbh_list):
            x, y, z = xyz
            writer.writerow({"id": idx, "x": x, "y": y, "z": z, "dbh": dbh})

    logger.info("CSV file", output_fp, "written.")

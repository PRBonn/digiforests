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
import numpy as np
from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.logging import logger

from .cylinder import Cylinder
from .ransac import RansacFitter
from .least_squares import LeastSquaresFitter


def fit_cylinders(
    cloud: Cloud, voxel_size: float, ransac_iterations=10000, lsq_iterations=100
) -> list[Cylinder]:
    cylinders = []
    instance = cloud.instance
    unique_iids = np.unique(instance)
    for iid in unique_iids:
        if iid < 0:
            continue
        logger.trace("Processing instance", iid, "/", len(unique_iids))
        iid_mask = instance == iid
        iid_cloud = cloud.select_by_mask(iid_mask.ravel())
        ransac_fitter = RansacFitter(
            iid_cloud, ransac_iterations=ransac_iterations, voxel_size=voxel_size
        )
        ransac_cylinder, ransac_inlier_ratio = ransac_fitter.fit()
        if ransac_cylinder is None:
            logger.trace("instance", iid, "ransac failed")
            continue

        lsq_fitter = LeastSquaresFitter(
            voxel_size=voxel_size,
            iterations=lsq_iterations,
            mode="analytic",
            verbose=True,
        )
        lsq_success, lsq_cylinder, lsq_inlier_ratio = lsq_fitter.fit(
            initial_guess=ransac_cylinder, points=iid_cloud.points
        )
        if lsq_success:
            if lsq_inlier_ratio > ransac_inlier_ratio:
                cylinders.append(lsq_cylinder)
            else:
                cylinders.append(ransac_cylinder)
        else:
            cylinders.append(ransac_cylinder)
        logger.trace("Instance id", iid, "cylinder estimated: ", cylinders[-1])
    return cylinders

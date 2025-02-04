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
from scipy.spatial import KDTree

from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.logging import logger

from ..utils import log_scalars_in_arguments


@log_scalars_in_arguments()
def normalize_height(
    non_ground_cloud: Cloud,
    ground_cloud: Cloud,
    k: int = 5,
    p: int = 2,
    rmax: float = 10.0,
) -> Cloud:
    """
    Normalizes non_ground_cloud using the 2d information (points) from
    ground_cloud. Runs Knn Inverse Distance Weighting (IDW).

    :param k: number of nearest neighbours to use for IDW.
    :param p: power factor for IDW.
    :param rmax: max distance knn queries.
    :return: the normalized version of non_ground_cloud.
    """
    search_xy_points = ground_cloud.points[:, :2]
    search_z = ground_cloud.points[:, 2]
    query_xy_points = non_ground_cloud.points[:, :2]
    # search for z of query xy points corresponding to search xyz
    logger.debug("Fitting kdtree to point cloud")
    kdtree = KDTree(data=search_xy_points)
    logger.debug("Fit complete")
    dist, idx = kdtree.query(
        query_xy_points, k=k, p=2, distance_upper_bound=rmax, workers=-1
    )
    logger.debug("query complete.")

    invalid_idx = np.flatnonzero(dist[:, 0] == np.inf)
    if invalid_idx.size:
        logger.debug(
            f"no nearest neighbour withing range for {invalid_idx.size} points,"
            " taking only closest neighbor for elevation"
        )
        patch_dist, patch_idx = kdtree.query(
            query_xy_points[invalid_idx], k=1, p=2, workers=-1
        )
        dist[invalid_idx, 0] = patch_dist
        idx[invalid_idx, 0] = patch_idx

    # dist can be 0. weight should be np.inf. raises a warning tho.
    epsilon = 1e-8
    non_zero_dist = dist
    non_zero_dist[dist == 0] += epsilon
    weights = 1 / np.power(non_zero_dist, p)
    weights[dist == 0] = np.inf
    clean_idx = idx.copy()
    # the dist for these values will be inf, so weights = 0
    clean_idx[idx == len(search_z)] = 0

    z_interp = np.sum(weights * search_z[clean_idx], axis=1) / np.sum(weights, axis=1)
    z_interp[dist[:, 0] == 0] = search_z[idx[dist[:, 0] == 0][:, 0]]
    logger.debug("Knn IDW interpolation complete.")
    # now actually normalize the height
    normalized_cloud = non_ground_cloud.clone()
    normalized_cloud.points[:, 2] = normalized_cloud.points[:, 2] - z_interp
    return normalized_cloud

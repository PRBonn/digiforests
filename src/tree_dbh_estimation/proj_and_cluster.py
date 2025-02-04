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
from quickshift import Quickshift

from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.logging import logger

from .utils import log_scalars_in_arguments


@log_scalars_in_arguments()
def project_2d_then_cluster_3d(
    cloud: Cloud,
    k: int = 150,
    beta: float = 0.5,
    epsilon: float = 0.0,
    knn_algo: str = "kd_tree",
    leaf_size: int = 30,
) -> Cloud:
    """
    Uses quickshift. First clusters cores on a 2d ground projection. Then clusters to cores in 3d.

    :param cloud: the cloud to cluster
    :param k: number of nearest neighbours
    :param beta: quickshift density parameter. 1. clusters everything as one, 0 every point as one.
    :param epsilon: unused
    :param knn_algo: kd_tree will suffice. alternative is ball_tree for high dimensional cases.
    :param leaf_size: kd_tree leaf size
    :returns: cloud with an instance attribute and colors colorized by instance.
    """
    qspp = Quickshift(
        k=k, beta=beta, epsilon=epsilon, knn_algo=knn_algo, leaf_size=leaf_size
    )
    cores = qspp.cluster_cores(cloud.points[:, :2])
    instance = qspp.cluster_to_cores(cloud.points, cores)
    # NOTE: qs currently returns -1 for noise points
    logger.info(f"point cloud has {len(np.unique(instance))} clusters")
    cluster_cloud = cloud.clone()
    cluster_cloud.instance = instance
    return cluster_cloud

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
import CSF
import numpy as np

from digiforests_dataloader.utils.cloud import Cloud

from ..utils import log_scalars_in_arguments


@log_scalars_in_arguments()
def segment_ground(
    cloud: Cloud,
    bSloopSmooth: float = False,
    cloth_resolution: float = 0.5,
    rigidness: int = 3,
    time_step: float = 0.65,
    class_threshold: float = 0.5,
    iterations: int = 500,
) -> tuple[Cloud, Cloud]:
    """
    Uses CSF

    :returns: ground cloud, non ground cloud
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = bSloopSmooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.interations = iterations

    csf.setPointCloud(cloud.points.astype(np.float64))
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground, exportCloth=False)

    ground_cloud = cloud.select_by_index(np.asarray(ground))
    non_ground_cloud = cloud.select_by_index(np.asarray(non_ground))
    return ground_cloud, non_ground_cloud

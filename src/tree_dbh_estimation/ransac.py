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
from digiforests_dataloader.utils.logging import bar

from .cylinder import Cylinder


class RansacFitter:
    def __init__(
        self,
        cloud: Cloud,
        voxel_size,
        ransac_iterations=10000,
        seed=None,
        max_radius=0.5,
    ):
        self.pcd = cloud.tpcd.to_legacy()
        self.indices = np.arange(len(self.pcd.points))
        self.ransac_iterations = ransac_iterations
        self.voxel_size = voxel_size
        self.rng = np.random.default_rng(seed)
        self.max_radius = max_radius

    def fit(self):
        best_inliers = 0
        best_estimate = None
        for _ in bar(
            range(self.ransac_iterations),
            desc="ransac iterations",
            total=self.ransac_iterations,
        ):
            sample_pcd = self.pcd.select_by_index(self.rng.choice(self.indices, size=5))
            inlier_points, cylinder = self.one_round(sample_pcd)
            if np.arccos(cylinder.axis[2]) >= np.pi / 8:
                continue
            num_inliers = np.sum(inlier_points)
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_estimate = cylinder

        if best_estimate is not None and best_estimate.radius > self.max_radius:
            return None, 0.0

        return best_estimate, best_inliers / len(self.indices)

    def one_round(self, pcd):
        cylinder = self.fit_cylinder(pcd)
        points = np.asarray(self.pcd.points)
        distances = cylinder.belong_to(points)
        inlier_points = np.abs(distances) < 0.1 * self.voxel_size
        return inlier_points, cylinder

    def fit_cylinder(self, pcd):
        mu, sigma = pcd.compute_mean_and_covariance()
        if np.linalg.det(sigma) < 1e-8:
            return Cylinder(np.zeros((3,)), 0.0, frame=np.eye(3))
        U, S, _ = np.linalg.svd(sigma)
        axis = U[:, 0]
        if axis[2] < 0.0:
            axis *= -1.0
        radius = np.sqrt(S[1])
        return Cylinder(center=mu, frame=U, radius=radius)

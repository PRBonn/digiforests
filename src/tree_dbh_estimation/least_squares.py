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
import copy

import numpy as np
import open3d as o3d
from tqdm import tqdm
from digiforests_dataloader.utils.logging import logger

from .cylinder import Cylinder


class LeastSquaresFitter:
    def __init__(
        self,
        voxel_size: float,
        iterations: int = 100,
        max_radius: float = 0.5,
        chi_square_threshold=1e-8,
        delta_x_norm_threshold=1e-5,
        mode="analytic",
        verbose=True,
    ) -> None:
        self.iterations = iterations
        self.voxel_size = voxel_size
        self.max_radius = max_radius
        self.chi_square_threshold = chi_square_threshold
        self.delta_x_norm_threshold = delta_x_norm_threshold
        self.mode = mode
        self.verbose = verbose

    def fit(self, initial_guess: Cylinder, points: np.ndarray):
        assert initial_guess is not None
        cylinder = copy.deepcopy(initial_guess)
        for iter in tqdm(
            range(self.iterations), desc="least sqaures", total=self.iterations
        ):
            cylinder, plussed_error, delta_x_norm = self.least_squares(cylinder, points)
            chi_square = calc_chi_square(plussed_error)
            if (
                chi_square < self.chi_square_threshold
                or delta_x_norm < self.delta_x_norm_threshold
            ):
                if self.verbose:
                    logger.trace("LSQ Iter: ", iter, " chi_square = ", chi_square)
                break

        distances = cylinder.belong_to(points)
        num_inliers = np.sum(np.abs(distances) < 0.1 * self.voxel_size)
        return (
            (cylinder.radius < self.max_radius) and self._check_axis(cylinder),
            cylinder,
            num_inliers / len(points),
        )

    def _check_axis(self, cylinder: Cylinder):
        return np.arccos(cylinder.axis[2]) < np.pi / 8

    def least_squares(self, state: Cylinder, points: np.ndarray):
        error = state.belong_to(points)
        gm_weights = gm_kernel_weights(error)
        if self.mode == "numeric":
            J = self.numerical_jacobian(state, points)
        else:
            J = self.analytic_jacobian(state, points)
        # H = J.T @ J + np.eye(7)
        H = (J.T * gm_weights.reshape(1, -1)) @ J + np.eye(7)

        # g = J.T @ error
        g = (J.T * gm_weights.reshape(1, -1)) @ error
        delta_x = np.linalg.solve(H, -g)
        plussed_state = self._box_plus(state, delta_x)
        plussed_error = plussed_state.belong_to(points)
        return plussed_state, plussed_error, np.linalg.norm(delta_x)

    def numerical_jacobian(self, state, points):
        __EPSILON__ = 1e-8
        __DIM__ = 7
        J = np.zeros((points.shape[0], __DIM__))
        cost = state.belong_to(points)
        for i in range(__DIM__):
            delta = np.zeros((__DIM__,))
            delta[i] = __EPSILON__
            plussed = self._box_plus(state, delta)
            cost_plussed = plussed.belong_to(points)
            J[:, i] = (cost_plussed - cost) / __EPSILON__
        return J

    def analytic_jacobian(self, state, points):
        mu = state.center
        axis = state.axis
        J = np.zeros((points.shape[0], 7))
        for idx in range(0, points.shape[0]):
            centered_point = points[idx, :] - mu
            cross = np.cross(axis, centered_point)
            norm = np.linalg.norm(cross)
            J_norm = cross / norm
            axis_skew = skew(axis)
            J[idx, :3] = J_norm @ -axis_skew
            J[idx, 3:6] = J_norm @ (skew(centered_point) @ axis_skew)
            J[idx, -1] = -1
        return J

    @staticmethod
    def _box_plus(x: Cylinder, delta_x):
        plussed_center = x.center + delta_x[:3].reshape(-1)
        if x.frame is not None:
            plussed_frame = (
                o3d.geometry.get_rotation_matrix_from_xyz(
                    [delta_x[3], delta_x[4].item(), delta_x[5].item()]
                )
                @ x.frame
            )
            plussed_axis = None
        else:
            plussed_axis = (
                o3d.geometry.get_rotation_matrix_from_xyz(
                    [delta_x[3], delta_x[4].item(), delta_x[5].item()]
                )
                @ x.axis
            )
            plussed_frame = None
        plussed_radius = x.radius + delta_x[-1].item()
        return Cylinder(
            center=plussed_center,
            radius=plussed_radius,
            frame=plussed_frame,
            axis=plussed_axis,
        )


def skew(vector: np.ndarray):
    assert len(vector) == 3
    v1, v2, v3 = vector
    _skew = np.zeros((3, 3))
    _skew = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
    return _skew


def calc_chi_square(error_vector):
    reshaped_error = error_vector.reshape(-1, 1)
    return reshaped_error.T @ reshaped_error


def gm_kernel_weights(error: np.ndarray):
    """geman mcclure robust kernel"""
    return 0.01**2 / ((0.01 + error**2) ** 2)

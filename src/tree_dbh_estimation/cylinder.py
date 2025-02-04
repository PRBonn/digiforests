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
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class Cylinder:
    def __init__(self, center, radius, axis=None, frame=None):
        self.center = center
        self.radius = radius
        if axis is not None:
            self.axis = axis
            if frame is not None:
                assert np.allclose(
                    self.axis, frame[:, 0]
                ), "first column of the frame should be the axis vector"
            self.frame = frame
        else:
            assert frame is not None, "pass either axis or frame"
            self.frame = frame
            self.axis = frame[:, 0]

    @property
    def pose(self):
        T = np.eye(4)
        T[:3, :3] = R.align_vectors(
            self.axis.reshape(1, 3), np.array([0, 0, 1]).reshape(1, 3)
        )[0].as_matrix()
        T[:3, -1] = self.center
        return T

    def __str__(self) -> str:
        return f"Cylinder with radius {self.radius}, center in {self.center}, axis = {self.axis}"

    def __repr__(self) -> str:
        return self.__str__()

    def belong_to(self, points):
        centered_points = points - self.center
        surface_normal = np.cross(self.axis, centered_points)
        per_point_cost = np.linalg.norm(surface_normal, axis=1) - self.radius
        return per_point_cost

    def to_open3d(self, height):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            self.radius,
            height=height,
        )
        return cylinder.transform(self.pose)

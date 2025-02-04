# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
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

import math
import random

from torch import Tensor
import torch
from ..utils.logging import logger

Number = int | float


class RandomRotate(object):
    """
    Rotates positions around a specific axis by a randomly sampled
    angle within a given interval.
    Best as a batch transform on the gpu.

    :params degrees: Rotation interval from which the rotation
    angle is sampled. If a number instead of a tuple, the interval is
    given by [-degrees, degrees].
    :param axis: The rotation axis. 0 - x, 1 - y, 2 - z.
    :param rotate_vectors: rotates any vector information like `normal` and
    `offset` if they exist (with those keys).
    """

    def __init__(
        self,
        degrees: tuple[Number] | Number = 180,
        axis: int = 2,
        rotate_vectors=True,
        minkowski=False,
        verbose=False,
    ):
        "minkowski batch is treated differently for pos or coords rotation"

        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.rotate_vectors = rotate_vectors
        self.minkowski = minkowski
        self.verbose = verbose

    def __call__(self, data: dict[str, Tensor]):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data["pos"].shape[-1] == 2:
            R_matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                R_matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                R_matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                R_matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        R_matrix = Tensor(R_matrix).to(data["pos"].dtype).to(data["pos"].device)

        if self.minkowski:
            data["pos"][:, 1:] = torch.matmul(data["pos"][:, 1:], R_matrix.T)
        else:
            data["pos"] = torch.matmul(data["pos"], R_matrix.T)

        if self.rotate_vectors and "normal" in data:
            data["normal"] = torch.matmul(data["normal"], R_matrix.T)
        if self.rotate_vectors and "offset" in data:
            data["offset"] = torch.matmul(data["offset"], R_matrix.T)

        if self.verbose:
            logger.trace("Random Rotate transform applied")

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.degrees}, axis={self.axis})"

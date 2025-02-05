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

import torch
import random
from torch import Tensor
from ..utils.logging import logger

Number = int | float


class RandomReflection(object):
    """
    Reflects positions
    Best as a batch transform on the gpu.

    :param axis: The rotation axis. 0 - x, 1 - y, 2 - z.
    :param random_reflect: if True, randomly decides whether to reflect or not
    :param reflect_vectors: reflects any vector information like `normal` and
    `offset` if they exist (with those keys).
    """

    def __init__(
        self,
        axis=2,
        random_reflect=True,
        reflect_vectors=True,
        minkowski=False,
        verbose=False,
    ):
        "minkowski batch is treated differently for pos or coords rotation"

        self.axis = axis
        self.random_reflect = random_reflect
        self.reflect_vectors = reflect_vectors
        self.minkowski = minkowski
        self.verbose = verbose

    def __call__(self, data: dict[str, Tensor]):
        if self.random_reflect and random.random() < 0.5:
            # no augmentation
            return data

        R_matrix = -1 * torch.eye(3).to(data["pos"].dtype).to(data["pos"].device)
        R_matrix[self.axis, self.axis] = 1

        if self.minkowski:
            data["pos"][:, 1:] = torch.matmul(data["pos"][:, 1:], R_matrix.T)
        else:
            data["pos"] = torch.matmul(data["pos"], R_matrix.T)

        if self.reflect_vectors and "normal" in data:
            data["normal"] = torch.matmul(data["normal"], R_matrix.T)
        if self.reflect_vectors and "offset" in data:
            data["offset"] = torch.matmul(data["offset"], R_matrix.T)

        if self.verbose:
            logger.trace("Random Reflect transform applied about axis", self.axis)

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(axis={self.axis})"

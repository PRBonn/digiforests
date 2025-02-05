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

import random
from torch import Tensor
from ..utils.logging import logger

Number = int | float
Number_Sequence = list[Number] | tuple[Number, Number]


class RandomUniformScale(object):
    """
    Scales positions by a randomly sampled factor `s` within a
    given interval.
    Also scales any vector information like normals and offsets if they
    exist.
    Best as a batch transform on the gpu.

    :param scales: interval [a, b] from which scale is randomly sampled.
    :param scale_vectors: scales any vector information like `normal` and
    `offset` if they exist (with those keys).
    """

    def __init__(
        self,
        scales: Number_Sequence,
        scale_vectors: bool = True,
        minkowski=False,
        verbose=False,
    ):
        "minkowski batch is treated differently for pos or coords rotation"
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.scale_vectors = scale_vectors
        self.minkowski = minkowski
        self.verbose = verbose

    def __call__(self, data: dict[str, Tensor]):
        scale = random.uniform(*self.scales)
        if self.minkowski:
            data["pos"][:, 1:] = data["pos"][:, 1:] * scale
        else:
            data["pos"] = data["pos"] * scale

        if self.scale_vectors and "normal" in data:
            raise NotImplementedError(
                "normals have some special consideration maybe. see deltaconv for weirdness"
            )
        if self.scale_vectors and "offset" in data:
            data["offset"] = data["offset"] * scale

        if self.verbose:
            logger.trace("Random Scale transform applied")
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.scales})"

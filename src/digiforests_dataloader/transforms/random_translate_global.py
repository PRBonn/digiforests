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
from torch import Tensor
from itertools import repeat

Number = int | float
Number_Sequence = list[Number] | tuple[Number]


class RandomTranslateGlobal(object):
    """
    Translates shapes by randomly sampled translation values
    within a given interval. This translation happens for the entire shape,
    retaining local relationships.
    Best as a batch transform on the gpu.

    :param translate: Maximum translation in each dimension, defining the
    range (-translate, +translate) per axis to sample from. If a number
    instead of a sequence, the same range is used for each dimension.
    """

    def __init__(self, translate: Number | Number_Sequence):
        self.translate = translate

    def __call__(self, data: dict[str, Tensor]):
        euc_dim = data["pos"].shape[-1]
        translate = self.translate
        if isinstance(translate, (int, float)):
            translate = list(repeat(translate, times=euc_dim))
        assert len(translate) == euc_dim

        transform = []
        for dim in range(euc_dim):
            transform.append(
                data["pos"]
                .new_empty(1)
                .uniform_(-abs(translate[dim]), abs(translate[dim]))
            )
        data["pos"] = data["pos"] + torch.stack(transform, dim=-1)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.translate})"

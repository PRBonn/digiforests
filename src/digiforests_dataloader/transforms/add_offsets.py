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
from ..utils.logging import logger


class AddOffsets(object):
    """
    Best as a pre_transform.
    NOTE: Batches are not handled.
    NOTE: the transformed data's offset can contain nan's when there are
    noise points which do not belong to any instance. these should be
    filtered in the loss.

    :param noise_iid: this instance id is ignored when calculating offsets.
    """

    def __init__(self, ignore_id: int = 0, verbose=True):
        self.ignore_id = ignore_id
        self.verbose = verbose

    def __call__(self, data: dict[str, torch.Tensor]):
        if self.verbose:
            logger.trace("applying transform - add offsets")
        pos = data["pos"]
        iids = data["instance"]
        offset = torch.full_like(pos, 0.0)
        unique_iids = iids.unique()
        unique_iids = unique_iids[unique_iids != self.ignore_id]

        for iid in unique_iids:
            iid_mask = (iids == iid).flatten()
            instance_pos = pos[iid_mask]
            instance_center = instance_pos.mean(dim=-2)
            instance_offset = instance_center - instance_pos
            offset[iid_mask] = instance_offset

        data["offset"] = offset
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_iid={self.noise_iid})"

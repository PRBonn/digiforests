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

from torch import nn, Tensor


class OffsetLoss(nn.L1Loss):
    def __init__(self, ignore_target_instance_id: int = 0) -> None:
        """
        division happens by number of valid instance indices in forward
        """
        super().__init__(reduction="sum")
        self.ignore_target_instance_id = ignore_target_instance_id

    def forward(
        self, input_offset: Tensor, target_offset: Tensor, target_instances: Tensor
    ):
        valid_mask = target_instances != self.ignore_target_instance_id
        if not valid_mask.sum():
            # TODO: hack to avoid nans
            return False
        valid_input = input_offset[valid_mask]
        valid_target = target_offset[valid_mask]
        res_loss_sum = super().forward(valid_input, valid_target)
        return res_loss_sum / valid_mask.sum()

    def __call__(
        self, input_offset: Tensor, target_offset: Tensor, target_instances: Tensor
    ):
        return self.forward(input_offset, target_offset, target_instances)

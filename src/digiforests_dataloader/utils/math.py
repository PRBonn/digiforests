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

from typing import Any

import numpy.typing as npt


def lmap(
    v: npt.NDArray,
    x: tuple[Any, Any] | npt.NDArray,
    y: tuple[Any, Any] | npt.NDArray,
):
    """Linear map of value(s) v with range x to desired range y."""
    # not adding an epsilon to prevent zero div, can add later
    assert x[1] != x[0], "Division by zero, check x range"
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def safe_divide(num, denom, eps: float = 1e-8):
    """prevents division by zero by clipping denom to eps"""
    if denom < eps:
        denom = eps
    return num / denom


def vector_safe_divide(num, denom, eps: float = 1e-8):
    """prevents division by zero by clipping denom to eps. vector version"""
    denom[denom < eps] = eps
    return num / denom

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
import functools
from pathlib import Path
from typing import Callable

from digiforests_dataloader.utils.logging import logger


def log_scalars_in_arguments(
    callable: Callable | None = None, *, level="debug", name=None
):
    """
    decorator to log the values of scalars in a functions arguments.
    usage:
    @log_scalars_in_arguments(level="debug", name="Function name")
    or @log_scalars_in_arguments

    :param callable: the callable to wrap
    :param level: info, debug, warning, error
    :param name: The description to use in a log giving execution time. Defaults to __name__ attribute.
    """

    def _decorate(callable):
        @functools.wraps(callable)
        def wrapped_callable(*args, **kwargs):
            log_str = ""
            first = True
            for arg in args:
                if isinstance(arg, (int, float, str, Path)):
                    log_str += ("" if first else ", ") + str(arg)
                    first = False
            for kw, v in kwargs.items():
                if isinstance(v, (int, float, str, Path)):
                    log_str += ("" if first else ", ") + f"{kw}: {str(v)}"
                    first = False

            if log_str:
                getattr(logger, level)(
                    f"{name if name is not None else callable.__name__} called with - {log_str}.",
                )

            value = callable(*args, **kwargs)
            return value

        return wrapped_callable

    if callable:
        return _decorate(callable)
    return _decorate

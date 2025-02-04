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

import os
import logging
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.theme import Theme

color_map = {
    "logging.level.trace": "steel_blue",
    "logging.level.debug": "cyan",
    "logging.level.info": "bold white",
    "logging.level.warning": "bold yellow",
    # 'SUCCESS': 'bold green',
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
}

log_level_mapping = {"trace": 5, "debug": 10, "info": 20, "warn": 30}
# https://github.com/Textualize/rich/issues/1161
console = Console(theme=Theme(color_map), markup=True)


# use progress bars
def bar(sequence: Iterable, desc: str = "Working...", total: int | None = None):
    """
    A straight replacement for tqdm.tqdm(Iterable, desc="", total=##)
    For when you want to print or log in between a progress bar.

    Usage:
        for i in bar(range(10)):
            print(i)

    """
    prog = Progress(console=console)
    if total is None:
        if hasattr(sequence, "__len__"):
            total = len(sequence)
    prog.start()
    yield from prog.track(sequence=sequence, total=total, description=desc)
    prog.stop()


# use progress bars
def get_progress_ctx():
    """
    Usage:
        with get_progress_ctx() as progress:
            total = 100
            bar = progress.add_task("[light_slate_blue]Building edge list...", total=total)
            for i in range(0, total):
                progress.update(bar, advance=1)

    :return: a Progress context manager
    """
    return Progress(console=console)


class Logger:
    """
    Just import logger and you're off with a default logger that can use
    info(), debug(), warning(), error(). But the name will be
    "pb_utils-logger".
    To customize this, at module initialization (or at some other point)
    call name_with() to name the logger.
    """

    def __init__(self):
        self._name = None
        self._logger = None
        pass

    def name_with(self, name: str):
        """
        Creates/sets the logger with the given name.
        If the logger already has a name, meaning it has been initialized,
        it is not overwritten.
        This is not a perfect or a good solution really. But I don't really
        want to engineer a full-blown logger class right now.
        So this will have to do. Not that the logger name really matters
        much.

        :param name: the name of the logger
        :type name: str
        :return: self
        """
        if self._name is None:
            self._name = name
            self._create()
        return self

    def _create(self):
        """
        Creates a logger. but if a logger with self._name has already been
        created, getLogger() should just return the previous one. the config
        for that logger will be redone i guess. overhead.
        """
        self._logger = logging.getLogger(self._name)
        logging.addLevelName(5, "TRACE")
        log_level_name = os.environ.get("LOG_LEVEL", "debug").lower()
        log_level = log_level_mapping.get(log_level_name, 5)
        self._logger.setLevel(5)
        rich_handler = RichHandler(
            level=log_level, rich_tracebacks=True, console=console, markup=True
        )
        formatter = self._create_formatter("{message}")
        self.add_handler(rich_handler, formatter=formatter)

    # TODO: This entire thing needs major cleanup
    def _create_formatter(self, format="[{asctime} - {levelname}] {name} - {message}"):
        formatter = logging.Formatter(
            format,
            datefmt="%d-%m-%y %X",
            style="{",
        )
        return formatter

    def add_file_handler(self, log_file_fp: Path, level: int = logging.DEBUG):
        file_handler = logging.FileHandler(log_file_fp.as_posix())
        file_handler.setLevel(level)
        self.add_handler(file_handler)

    def add_handler(self, handler, format: bool = True, formatter=None):
        """
        if format and no formatter is given, a default formatter is used
        """
        if format:
            formatter = self._create_formatter() if not formatter else formatter
            handler.setFormatter(formatter)

        self._logger.addHandler(handler)

    def _check_existence(self):
        """
        if a logger doesnt exist, creates a default one.
        """
        if self._logger is None:
            self.name_with("pb_utils-logger")

    def trace(self, *args):
        """stacklevel is 2 to catch the name of the caller"""
        self._check_existence()
        message = _string_from_args(*args)
        return self._logger.log(5, message, stacklevel=2)

    def debug(self, *args):
        """stacklevel is 2 to catch the name of the caller"""
        self._check_existence()
        message = _string_from_args(*args)
        return self._logger.debug(message, stacklevel=2)

    def info(self, *args):
        """stacklevel is 2 to catch the name of the caller"""
        self._check_existence()
        message = _string_from_args(*args)
        return self._logger.info(message, stacklevel=2)

    def warning(self, *args):
        """stacklevel is 2 to catch the name of the caller"""
        self._check_existence()
        message = _string_from_args(*args)
        return self._logger.warning(message, stacklevel=2)

    def error(self, *args):
        """stacklevel is 2 to catch the name of the caller"""
        self._check_existence()
        message = _string_from_args(*args)
        return self._logger.error(message, stacklevel=2)


def _string_from_args(*args):
    message = ""
    for arg in args:
        message = message + str(arg) + " "
    return message[:-1]


logger = Logger()
logger.name_with("digiforests")

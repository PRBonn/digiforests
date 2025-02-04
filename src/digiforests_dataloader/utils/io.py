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

import json
from pathlib import Path

import yaml

from .logging import logger
from .serialize import PathEncoder


def load_yaml(fp: Path):
    """load yaml file at fp"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    with fp.open("r") as yaml_file:
        res = yaml.safe_load(yaml_file)
    return res


def load_json(fp: Path):
    """load json file at fp"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    with fp.open("r") as json_file:
        res = json.load(json_file)
    return res


def write_yaml(obj, fp: Path, overwrite=False, verbose=False):
    """
    write obj to yaml file at fp

    :param overwrite: if not True and file already exists, exception raised
    :param verbose: logs with level info
    """
    if fp.exists() and not overwrite:
        raise Exception(f"{fp} already exists")
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as yaml_file:
        yaml.safe_dump(obj, yaml_file, default_flow_style=False)
        if verbose:
            logger.info(f"{fp} written.")


def write_json(
    obj,
    fp: Path,
    indent: int = 4,
    sort_keys=True,
    overwrite=False,
    verbose=False,
    encoder=PathEncoder,
    **kwargs,
):
    """
    write obj to json file at fp. any extra kwargs are passed along to
    json.dump

    :param indent: default indentation for prettiness
    :param sort_keys: default sort for prettiness
    :param overwrite: if not True and file already exists, exception raised
    :param verbose: logs with level info
    :param encoder: by default use the PathEncoder, since we use paths a lot
    """
    if fp.exists() and not overwrite:
        raise Exception(f"{fp} already exists")
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as json_file:
        json.dump(
            obj,
            json_file,
            indent=indent,
            sort_keys=sort_keys,
            cls=PathEncoder,
            **kwargs,
        )
        if verbose:
            logger.info(f"{fp} written.")


def load_yaml_or_json(fp: Path):
    """loads file at fp based on extension"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    if fp.suffix == ".yaml" or fp.suffix == ".yml":
        return load_yaml(fp)
    elif fp.suffix == ".json":
        return load_json(fp)
    else:
        raise Exception(f"check the extension for {fp}")

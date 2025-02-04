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
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

from digiforests_dataloader.utils.logging import logger


class ConsoleLogger(Logger):
    def __init__(self, level="trace", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.level = level

    @property
    def name(self):
        return "ConsoleLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        getattr(logger, self.level)(f"Hyperparameters: {json.dumps(params, indent=4)}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        getattr(logger, self.level)(
            f"Step: {step}, Metrics: {json.dumps(metrics, indent=4)}"
        )


def sync_config_keys(
    configs: list[dict],
    keys: list[str],
    recursive: bool = True,
    raise_on_mismatch: bool = True,
    overwrite: bool = False,
):
    """
    checks every config dict for each key in key and makes sure the
    values are the same. if recursive, checks for any dicts within dicts,
    and requires all keys have the same value

    :param configs: list of dictionaries or configs. does not check recursively
    :param keys: list of keys whose values should be same, example batch_size
    :param raise_on_mismatch: whether to raise an exception
    :param overwrite: whether to overwrite keys, if not raising. unimplemented.
    """
    if overwrite:
        raise NotImplementedError()

    def value_check_loop(master_value, key_value, raise_on_mismatch: bool = True):
        # master not defined
        if master_value == "nd":
            master_value = key_value
        # master is defined, key can be not defined
        if key_value == "nd":
            # key not defined
            # master can be defined or not, dont mess with anything
            return master_value
        # key is defined, and master should be same as key
        if master_value != key_value:
            if raise_on_mismatch:
                raise Exception(
                    f"master value {master_value} is not same as key value {key_value}."
                )
            else:
                raise NotImplementedError
        return master_value

    def recursive_get_dicts(dict_: dict):
        """
        gets all dicts that are values of keys in dict_. recursive therein.
        return is a list of dicts. will include the original dict.
        """
        ret_list = [dict_]
        for key, value in dict_.items():
            if isinstance(value, dict):
                ret_list.extend(recursive_get_dicts(value))
        return ret_list

    for key in keys:
        master_value = "nd"
        for config in configs:
            if recursive:
                dicts_to_check = recursive_get_dicts(config)
            else:
                dicts_to_check = [config]

            for check_dict in dicts_to_check:
                current_value = check_dict.get(key, "nd")
                try:
                    master_value = value_check_loop(
                        master_value, current_value, raise_on_mismatch=raise_on_mismatch
                    )
                except Exception as e:
                    logger.error(f"Error in configuration, for key {key}: {e}")
                    import sys

                    sys.exit(1)

# Copyright (c) Facebook, Inc. and its affiliates.
# List serialization code adopted from
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py

import torch
import pickle
import numpy as np
from pathlib import Path
from json import JSONEncoder


class PathEncoder(JSONEncoder):
    """
    handle pathlib.Path as well.

    check json.JSONEncoder docs for normal argument details.
    Pass any specific params here on to json.dump or json.dumps, the kwargs
    get used for Encoder initialization

    usage: json.dump[s](obj, [fp], cls=PathEncoder, [relative_to=Path(".")])

    :param relative_to: on dump, the written paths can be made relative
    to this path.
    """

    def __init__(self, relative_to: Path | None = None, *args, **kwargs):
        self.relative_to = relative_to
        super().__init__(*args, **kwargs)

    def default(self, o):
        """
        has to return the object itself and not call super.default(o).
        super.default raises an error
        """
        if isinstance(o, Path):
            if self.relative_to is not None:
                return o.relative_to(self.relative_to).as_posix()
            return o.as_posix()
        return super.default(o)


# from https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/79897b26a2c4185a3ed086f18be5ea300913d5b7/serialize.py
# why? see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
# https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader
# https://github.com/Lightning-AI/lightning/issues/17257#issuecomment-1493834663
class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)

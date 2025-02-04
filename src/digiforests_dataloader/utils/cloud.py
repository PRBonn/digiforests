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

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from .logging import logger
from .math import lmap
from typing import Callable, Any


class Cloud:
    """A wrapper around o3d.t.geometry.PointCloud."""

    UNLABELLED_POINT = -3
    MIN_VALID_INSTANCE = 0

    def __init__(self):
        """Use a class method instead."""
        super().__init__()
        self.tpcd = None

    @classmethod
    def from_file(cls, filename: str | Path) -> Cloud:
        """Construct from a file."""
        filename = Path(filename)
        assert filename.exists(), f"{filename} doesnt exist"
        tpcd = o3d.t.io.read_point_cloud(str(filename))
        cloud = cls()
        cloud.tpcd = tpcd
        return cloud

    # alias
    load = from_file

    @classmethod
    def from_points(cls, points: np.ndarray) -> Cloud:
        """Construct from points.

        :param points: np.ndarray
        """
        cloud = cls()
        cloud.tpcd = o3d.t.geometry.PointCloud(points)
        return cloud

    @classmethod
    def from_o3d(
        cls, o3d_tensor_point_cloud: o3d.t.geometry.PointCloud, clone=True
    ) -> Cloud:
        """Construct (optionally clones) from an o3d object.

        :param o3d_tensor_point_cloud: o3d.t.geometry.PointCloud:
        :param clone: (Default value = True)
        """
        cloud = cls()
        if clone:
            cloud.tpcd = o3d_tensor_point_cloud.clone()
        else:
            cloud.tpcd = o3d_tensor_point_cloud
        return cloud

    @classmethod
    def from_las(
        cls,
        filename: str | Path,
        zero_min: bool = False,
        custom_attribute_callback: Callable[[Any, Cloud], None] | None = None,
    ) -> Cloud:
        """
        Construct from a las file.

        :param zero_min: substracts min x, y, z from all points
        :param custom_attribute_callback: gets the LasData object as the first parameter
        and the Cloud object (with points) being constructed as the second. Arbitrary
        attributes be added to the cloud using Cloud.add_attribute()
        """
        filename = Path(filename)
        assert filename.exists(), f"{filename} doesnt exist"

        import laspy

        las = laspy.read(str(filename))
        logger.trace(f"las read from {filename}")

        xyz = las.xyz
        if zero_min:
            xyz = xyz - np.min(xyz, axis=0)
        points = xyz

        cloud = Cloud.from_points(points)

        if custom_attribute_callback is not None:
            custom_attribute_callback(las, cloud)

        return cloud

    # alias
    load_las = from_las

    @classmethod
    def from_array(
        cls,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        instance: np.ndarray | None = None,
        semantics: np.ndarray | None = None,
    ) -> Cloud:
        """Construct from arrays.

        :param points: np.ndarray:
        :param colors: np.ndarray | None:  (Default value = None)
        :param instance: np.ndarray | None:  (Default value = None)
        :param semantics: np.ndarray | None:  (Default value = None)
        """
        cloud = cls.from_points(points)
        if colors is not None:
            assert colors.ndim == 2
            assert colors.shape[1] == 3
            assert colors.shape[0] == cloud.num_points
            cloud.colors = colors.astype(float)
        if instance is not None:
            instance = instance.reshape((-1, 1))
            assert instance.shape[0] == cloud.num_points
            cloud.instance = instance.astype(np.int32)
        if semantics is not None:
            semantics = semantics.reshape((-1, 1))
            assert semantics.shape[0] == cloud.num_points
            cloud.semantics = semantics.astype(np.int32)

        assert cloud.check_state(), "the point cloud is malformed. debug"
        return cloud

    def clone(self) -> Cloud:
        """returns a clone."""
        return self.from_o3d(self.tpcd)

    def add_attribute(self, name: str, value: o3c.Tensor | np.ndarray, type=None):
        """Needs a type in case the data is ill formated.

        happens with las. if an ndarray, the value will be reshaped to
        (-1, 1) and then length must be equal to the number of points.

        :param name: str:
        :param value: o3c.Tensor | np.ndarray:
        :param type: (Default value = None)

        """

        if isinstance(value, o3c.Tensor):
            setattr(self.tpcd.point, name, value)
        elif isinstance(value, np.ndarray):
            assert type is not None, "type argument is required for np arrays"
            # attributes other than positions, colors, normals must be
            # of shape (num_points, 1). for writing to a pcd at least. this
            # may be different if you switch to ply as an output format
            value = value.reshape((-1, 1))
            assert value.shape[0] == self.num_points
            setattr(self.tpcd.point, name, o3c.Tensor(value.astype(type)))
        else:
            raise Exception("value was improper type.")
        return self

    def get_attribute(self, name):
        """
        :param name:

        """
        return getattr(self.tpcd.point, name).numpy()

    @property
    def points(self) -> np.ndarray:
        """Returns a shared memory array to the tensor positions."""
        return self.tpcd.point.positions.numpy()  # type: ignore

    @property
    def colors(self) -> np.ndarray:
        """Returns a shared memory array to the tensor colors."""
        return self.tpcd.point.colors.numpy()  # type: ignore

    @colors.setter
    def colors(self, colors_: np.ndarray):
        """Set the colors.

        copies the array

        :param colors_: np.ndarray:
        """
        assert isinstance(colors_, np.ndarray)
        self.tpcd.point.colors = o3c.Tensor(colors_)
        assert (
            self.check_state()
        ), f"improper state, {self.num_points} points and color tensor is {self.colors.shape}"

    @property
    def instance(self) -> np.ndarray:
        """Returns a shared memory array to the tensor instance."""
        return self.tpcd.point.instance.numpy()  # type: ignore

    @instance.setter
    def instance(self, instance_: np.ndarray):
        """Set the instance.

        copies the array

        :param instance_: np.ndarray:
        """
        assert isinstance(instance_, np.ndarray)
        self.tpcd.point.instance = o3c.Tensor(instance_.reshape(-1, 1))
        assert (
            self.check_state()
        ), f"improper state, {self.num_points} points and instance tensor is {self.instance}"

    @property
    def semantics(self) -> np.ndarray:
        """Returns a shared memory array to the tensor semantics."""
        return self.tpcd.point.semantics.numpy()  # type: ignore

    @semantics.setter
    def semantics(self, semantics_: np.ndarray):
        """Set the semantics.

        copies the array

        :param semantics_: np.ndarray:
        """
        assert isinstance(semantics_, np.ndarray)
        self.tpcd.point.semantics = o3c.Tensor(semantics_.reshape(-1, 1))
        assert (
            self.check_state()
        ), f"improper state, {self.num_points} points and semantics tensor is {self.semantics}"

    def select_by_index(self, indices, invert=False, remove_duplicates=False):
        """Select points and associated attributes by index.

        wrapper around the o3d api

        :param indices:
        :param invert: (Default value = False)
        :param remove_duplicates: (Default value = False)
        """
        return self.from_o3d(
            self.tpcd.select_by_index(indices, invert, remove_duplicates), clone=False
        )

    def select_by_mask(self, boolean_mask: o3c.Tensor, invert=False):
        """Select points and associated attributes by mask of shape {n,}

        wrapper around the o3d api

        :param boolean_mask: o3c.Tensor:
        :param invert: (Default value = False)
        :param boolean_mask: o3c.Tensor:
        """
        return self.from_o3d(
            self.tpcd.select_by_mask(boolean_mask, invert), clone=False
        )

    def paint_uniform_color(self, color: list):
        """wrapper around the o3d function.

        :param color: listlike of 3 floats [0,1] or ints
        """
        return self.from_o3d(self.tpcd.paint_uniform_color(color), clone=False)

    @property
    def has_colors(self) -> bool:
        """ """
        try:
            return hasattr(self.tpcd.point, "colors")
        except KeyError:
            # the tensor map raises a key error
            return False

    @property
    def has_instance(self) -> bool:
        """ """
        try:
            return hasattr(self.tpcd.point, "instance")
        except KeyError:
            # the tensor map raises a key error
            return False

    @property
    def has_semantics(self) -> bool:
        """ """
        try:
            return hasattr(self.tpcd.point, "semantics")
        except KeyError:
            # the tensor map raises a key error
            return False

    def voxel_down_sample(self, voxel_size: float):
        """Downsample the point cloud by a factor of voxel_size.

        Wrapper around the o3d method. how does it downsample? no clue.
        but i check the instance attribute returns, and it seems it
        picks a random point from the voxel for the attributes?

        :param voxel_size: float:
        """
        logger.debug(f"Downsampling point cloud with voxel_size {voxel_size}")
        return self.from_o3d(self.tpcd.voxel_down_sample(voxel_size), clone=False)

    def check_state(self) -> bool:
        """checks if the lengths of the points, colors and semantics are the
        same.

        checks if points and colors are of shape (n,3). checks if semantics
        is of shape (n,)
        """
        # short circuit some flags in case there are no colors or semantics
        flags = [
            self.points.ndim == 2,
            self.points.shape[1] == 3,
            (not self.has_colors) or self.colors.ndim == 2,
            (not self.has_colors) or self.colors.shape[1] == 3,
            (not self.has_colors) or self.points.shape[0] == self.colors.shape[0],
            (not self.has_instance) or self.instance.ndim == 2,
            (not self.has_instance) or self.points.shape[0] == self.instance.shape[0],
            (not self.has_instance) or self.instance.shape[1] == 1,
            (not self.has_semantics) or self.semantics.ndim == 2,
            (not self.has_semantics) or self.points.shape[0] == self.semantics.shape[0],
            (not self.has_semantics) or self.semantics.shape[1] == 1,
        ]
        return all(flags)

    @property
    def num_points(self) -> int:
        """ """
        return int(self.points.shape[0])

    def write(self, output_path: Path, verbose: bool = True) -> bool:
        """write cloud to disk.
        NOTE: to read custom attributes in CloudCompare, you'll need to
        take care of the types of attributes. ply doesnt support 64 bit ints.
        plus there was some issue with missed attributes with .pcd too.

        :param output_path: the path of the file
        :param verbose: bool:  (Default value = True)
        :returns: if the write was successful or not

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success: bool = o3d.t.io.write_point_cloud(
            str(output_path), self.tpcd, print_progress=True
        )
        if success:
            if verbose:
                logger.debug(f"Wrote the cloud file to disk at {output_path}")
        else:
            logger.error(f"Failed to save the cloud file to disk at {output_path}")

        return success

    def colorize_using_attribute(
        self,
        attr: str = "instance",
        min_valid_attr=None,
        random=False,
        blacken_noise=True,
        index_callback=lambda idx, elem: elem[0],
        custom_cmap=None,
    ) -> Cloud:
        """generate colors from unique values of an attribute.

        :param cloud: the point cloud to generate colors for
        :param attr: the attribute to generate colors for (Default value = "instance")
        :param min_valid_attr: the minimum value of the attribute. if the attribute is 'instance',
        defaults to MIN_VALID_LABEL of the cloud. Needs to be passed otherwise.
        :param random: generates a random color per label. (Default value = False)
        :param blacken_noise: noise (attr < min_valid_attr) is black (Default value = True)
        :param index_callback: Useful when the attr is not an Nx1 array. index_callback is used in something similar to
        for idx, attr_elem in enumerate(attr): yield index_callback(idx, attr_elem) and the value yielded needs to be hashable
        (to have a unique map of colors). The default callback assumes that attr is an Nx1 array, therefore the yielded value is just attr_elem (1x1) indexed with 0.
        :returns: self
        """
        attr_val = self.get_attribute(attr)
        if min_valid_attr is None:
            if attr == "instance":
                min_valid_attr = self.MIN_VALID_INSTANCE
            else:
                raise ValueError("need min_valid_attr argument")
        logger.debug(
            f"Colorizing with {attr}, {len(np.unique(attr_val))} unique values"
        )

        if random:
            random_colors = np.random.rand(len(np.unique(attr_val)), 3)
            attr_color_map = dict(zip(np.unique(attr_val), random_colors))
            if blacken_noise:
                colors = np.array(
                    [
                        (
                            attr_color_map[index_callback(idx, attr_elem)]
                            if attr_elem >= min_valid_attr
                            else [0.0, 0.0, 0.0]
                        )
                        for idx, attr_elem in enumerate(attr_val)
                    ]
                )
            else:
                colors = np.array([attr_color_map[attr_elem] for attr_elem in attr_val])
        else:
            import matplotlib.pyplot as plt

            if custom_cmap is not None:
                color_map = custom_cmap
            else:
                color_map = plt.get_cmap("tab20b")
            color_map.set_under(color="k")
            min_attr = np.min(attr_val)
            if blacken_noise:
                min_attr = min_valid_attr  # semantics under will get black
            max_attr = np.max(attr_val)
            max_attr = max_attr + 1 if max_attr == min_attr else max_attr
            colors = color_map(
                lmap(attr_val.reshape(-1), (min_attr, max_attr), (0, 1))
            )[:, :3]

        self.colors = colors
        return self

    def __str__(self) -> str:
        return str(self.tpcd)

    def __repr__(self) -> str:
        return self.__str__()

    def transform(self, transformation: np.ndarray) -> Cloud:
        transformed_tpcd = self.tpcd.transform(transformation)
        return Cloud.from_o3d(transformed_tpcd)

    def __add__(self, other: Cloud) -> Cloud:
        return Cloud.from_o3d(self.tpcd + other.tpcd)

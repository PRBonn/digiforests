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
import typer
import numpy as np
import open3d as o3d
from enum import Enum
from tqdm import tqdm
from pathlib import Path

from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.logging import logger, bar

__VOXEL_SIZE__ = 0.01


class Labels:
    class From(Enum):
        # subsemantic_labels
        STEM = 1
        CANOPY = 2
        # semantic_label
        TREE = 3

    class To(Enum):
        STEM = 5
        CANOPY = 6

    def __init__(self, semantics: np.ndarray, instance: np.ndarray) -> None:
        self.semantics = semantics
        self.instance = instance


def sanitize_key(key_list):
    key_list = [str(int(item)) for item in key_list]  # remove leading zeros
    key = "_".join(key_list)
    return key


def read_scans(scan_folder: Path) -> dict[str, Cloud]:
    scans = {}
    for fp in tqdm(sorted(scan_folder.glob("*.pcd")), "reading scans"):
        key = sanitize_key(fp.stem.split("_")[-2:])
        scans[key] = Cloud.load(fp)
    return scans


def read_labels(label_folder: Path) -> dict[str, Labels]:
    def read_single_label_file(fp: Path) -> Labels:
        data = np.fromfile(fp, dtype=np.uint32)
        # bits 0-7 correspond to semantic class
        semantic_labels = data & 0xFF
        print("unique semantics", np.unique(semantic_labels))
        tree_mask = semantic_labels == Labels.From.TREE.value
        # bits 8-15 correspond to subtree class
        subsemantic_labels = (data >> 8) & 0xFF
        print("unique subsemantics", np.unique(subsemantic_labels))
        stem_mask = np.logical_and(
            tree_mask, subsemantic_labels == Labels.From.STEM.value
        )
        canopy_mask = np.logical_and(
            tree_mask, subsemantic_labels == Labels.From.CANOPY.value
        )
        semantic_labels[stem_mask] = Labels.To.STEM.value
        semantic_labels[canopy_mask] = Labels.To.CANOPY.value
        print("unique semantics", np.unique(semantic_labels))
        print("-------")
        #
        # bits 16-31 correspond to instance
        instance_labels = data >> 16
        return Labels(semantic_labels, instance_labels)

    labels = {}
    for fp in tqdm(sorted(label_folder.glob("*.label")), "reading labels"):
        key = sanitize_key(fp.stem.split("_")[-2:])
        labels[key] = read_single_label_file(fp)
    return labels


def read_poses(pose_file: Path) -> dict[str, np.ndarray]:
    poses = {}
    with pose_file.open("r") as file:
        next(file)  # skip the header line
        for line in file:
            data = line.strip().split(", ")
            sec_nsec = f"{int(data[1])}_{int(data[2])}"
            pose = np.eye(4)
            pose[0, -1] = float(data[3])
            pose[1, -1] = float(data[4])
            pose[2, -1] = float(data[5])
            pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
                [
                    float(data[9]),
                    float(data[6]),
                    float(data[7]),
                    float(data[8]),
                ]
            )
            poses[sec_nsec] = pose
    return poses


def combine_scan_and_labels(
    scans: dict[str, Cloud],
    labels: dict[str, Labels],
):
    for key, scan in scans.items():
        scan.add_attribute("semantics", labels[key].semantics, type=np.int32)
        scan.add_attribute("instance", labels[key].instance, type=np.int32)
    return


def transform_scans(
    scans: dict[str, Cloud], scan_poses: dict[str, np.ndarray]
) -> list[Cloud]:
    transformed_scans = []
    missing_keys = 0
    for key, scan in bar(scans.items(), "transforming scans"):
        if key not in scan_poses:
            logger.warning(f"scan key {key} not found in poses. skipping")
            missing_keys += 1
            continue
        trans_scan = scan.transform(scan_poses[key])
        transformed_scans.append(trans_scan)
    logger.debug(f"{missing_keys} scans were skipped because of missing keys.")
    return transformed_scans


def aggregate_scans(scans: list[Cloud]) -> Cloud:
    agg_points = [scans[0].points]
    agg_sem = [scans[0].semantics]
    agg_inst = [scans[0].instance]
    for scan in tqdm(scans[1:], "aggregating scans"):
        agg_points.append(scan.points)
        agg_sem.append(scan.semantics)
        agg_inst.append(scan.instance)

    agg_points = np.concatenate(agg_points, axis=0)
    agg_sem = np.concatenate(agg_sem, axis=0)
    agg_inst = np.concatenate(agg_inst, axis=0)
    agg_cloud = Cloud.from_array(
        points=agg_points, semantics=agg_sem, instance=agg_inst
    )
    return agg_cloud


def denoise_cloud(cloud: Cloud, voxel_size: float):
    original_point_count = cloud.num_points
    o3d_cloud, _ = cloud.tpcd.remove_duplicated_points()
    vds_cloud = o3d_cloud.voxel_down_sample(voxel_size)
    stat_cloud, _ = vds_cloud.remove_statistical_outliers(20, 1.5)
    denoised_cloud = Cloud.from_o3d(stat_cloud)
    denoised_point_count = denoised_cloud.num_points
    logger.info(
        f"Point count - Original cloud: {original_point_count}, Denoised cloud: {denoised_point_count}, reduction - {(original_point_count - denoised_point_count)*100/original_point_count}%"
    )
    return denoised_cloud


def aggregate(
    exp_folder: Path = typer.Argument(
        ..., help="Path to the experiment folder containing scan and pose data."
    ),
    output_folder: Path = typer.Argument(
        ..., help="Path to the output folder where the aggregated cloud will be saved."
    ),
    denoise: bool = typer.Option(
        True, help="Apply denoising to the aggregated point cloud if set to True."
    ),
    voxel_down_sample_size: float = typer.Option(
        __VOXEL_SIZE__, help="Voxel down-sampling size for denoising the point cloud."
    ),
):
    """
    Aggregate individual point clouds and labels into a single PLY file.

    This function processes a DigiForests experiment folder, combining multiple scans
    and their corresponding labels into a unified point cloud representation.

    Workflow:
    1. Read pose information and individual scans
    2. Combine scans with their semantic and instance labels
    3. Transform scans to a common coordinate frame
    4. Aggregate all transformed scans
    5. Optionally denoise the aggregated cloud
    6. Save the result as a PLY file

    Args:
        exp_folder: Path to the experiment folder containing:
            - 'poses.txt': Scan pose information
            - 'individual_clouds/': Directory with individual point cloud scans
            - 'labels/': Directory with corresponding label files
        output_folder: Directory where the aggregated cloud will be saved
        denoise: If True, apply statistical outlier removal and voxel down-sampling
        voxel_down_sample_size: Voxel size for down-sampling during denoising

    Output:
        - Writes 'aggregated_cloud.ply' to the specified output folder

    Note:
        - Denoising can significantly reduce the point count but improve quality
        - The voxel down-sampling size affects the trade-off between detail and file size
    """
    scan_poses = read_poses(exp_folder / "poses.txt")
    scans = read_scans(exp_folder / "individual_clouds")
    labels = read_labels(exp_folder / "labels")
    combine_scan_and_labels(scans, labels)
    transformed_scans = transform_scans(scans, scan_poses)
    aggregated_cloud = aggregate_scans(transformed_scans)
    if denoise:
        denoised_cloud: Cloud = denoise_cloud(aggregated_cloud, voxel_down_sample_size)
        denoised_cloud.write(output_folder / "aggregated_cloud.ply")
    else:
        aggregated_cloud.write(output_folder / "aggregated_cloud.ply")


if __name__ == "__main__":
    typer.run(aggregate)

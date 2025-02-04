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
import re
import csv
import typer
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import KDTree

from digiforests_dataloader.utils.cloud import Cloud
from digiforests_dataloader.utils.render import render
from digiforests_dataloader.utils.logging import logger

from tree_dbh_estimation import fit_cylinders
from tree_dbh_estimation.cylinder import Cylinder
from tree_dbh_estimation import project_2d_then_cluster_3d
from tree_dbh_estimation.preprocess import normalize_height, clip_z

__VOXEL_SIZE__ = 0.05
__GROUND_CONF_THRESHOLD__ = 0.95
__STEM_CONF_THRESHOLD__ = 0.8


def read_label(fp: Path):
    binary_data = np.fromfile(fp, dtype=np.uint32)
    # bits 0-7 correspond to semantic class
    semantics = binary_data & 0xFF
    # bits 8-15 correspond to sem conf
    sem_conf = (binary_data >> 8) & 0xFF
    sem_conf = (sem_conf).astype(float) / 255
    # instance bits 16-31
    instance = binary_data >> 16

    return semantics, instance, sem_conf


def read_poses(slam_poses_file: Path) -> list[dict]:
    slam_poses = []
    with slam_poses_file.open("r") as poses_csv:
        csv_reader = csv.DictReader(poses_csv, skipinitialspace=True)
        for row in csv_reader:
            slam_poses.append(
                {
                    "sec": int(row["sec"]),
                    "nsec": int(row["nsec"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                }
            )
    return slam_poses


def transform_cloud(cloud, pose):
    transformation = np.eye(4)
    transformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        [
            pose["qw"],
            pose["qx"],
            pose["qy"],
            pose["qz"],
        ]
    )
    transformation[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    cloud.transform(transformation)
    return cloud


def aggregate_cloud(input_folder: Path, label_folder: Path | None):
    aggregated_points = []
    aggregated_semantics = []
    aggregated_instance = []
    aggregated_conf = []
    slam_poses = read_poses(input_folder / "poses.txt")
    for idx, pose in tqdm(
        enumerate(slam_poses), total=len(slam_poses), desc="aggregating clouds"
    ):
        cloud_filename = f"cloud_{pose['sec']:010d}_{pose['nsec']:09d}.pcd"
        cloud_path = input_folder / "individual_clouds" / cloud_filename
        label_folder = label_folder or (input_folder / "inference_labels")
        label_path = (label_folder / cloud_filename).with_suffix(".label")

        if cloud_path.exists():
            assert label_path.exists(), f"{label_path} doesnt exist for {cloud_path}"
            cloud = o3d.io.read_point_cloud(cloud_path.as_posix())
            cloud = transform_cloud(cloud, pose)
            semantics, instance, sem_conf = read_label(label_path)
            aggregated_points.append(np.asarray(cloud.points))
            aggregated_semantics.append(semantics)
            aggregated_instance.append(instance)
            aggregated_conf.append(sem_conf)
        else:
            logger.warning(f"{idx}: {cloud_path} does not exist.")
    aggregated_points = np.concatenate(aggregated_points, axis=0)
    aggregated_semantics = np.concatenate(aggregated_semantics, axis=0)
    aggregated_instance = np.concatenate(aggregated_instance, axis=0)
    aggregated_conf = np.concatenate(aggregated_conf, axis=0)
    aggregated_cloud = Cloud.from_array(
        points=aggregated_points,
        semantics=aggregated_semantics,
        instance=aggregated_instance,
    )
    aggregated_cloud.add_attribute("sem_conf", aggregated_conf, type=np.float32)
    return aggregated_cloud


def get_conf_mask(cloud: Cloud, conf: float):
    sem_conf = cloud.get_attribute("sem_conf")
    sem_conf_mask = sem_conf >= conf
    sem_conf_mask = sem_conf_mask.ravel()
    return sem_conf_mask


def extract_plot_id_from_path(input_path: Path) -> str:
    pattern = r"(exp\d{2}-[a-zA-Z]\d)"
    match = re.search(pattern, input_path.as_posix())
    if match:
        return match.group(1).lower()
    else:
        raise Exception(f"could not process path {input_path} for plot id")


def get_tree_id_to_xyz_dbh(inventory_csv: Path, plot_id: str, tree_locations_csv: Path):
    plot_key = plot_id.split("-")[1]
    df = pd.read_csv(
        inventory_csv.as_posix(),
        converters={"Plot": str.lower},
    )
    df["DBH [cm]"] = df["DBH [cm]"].str.replace(",", ".").astype(float)
    df_subset = df[["Plot", "Tree", "DBH [cm]"]]
    filtered_dbh_df = df_subset[df_subset["Plot"] == plot_key.lower()]

    tree_locations_df = pd.read_csv(tree_locations_csv.as_posix())
    tree_locations_dict = (
        tree_locations_df.set_index("tag_id")
        .apply(lambda row: (row["x"], row["y"], row["z"]), axis=1)
        .to_dict()
    )
    tree_dict = {}
    for index, row in filtered_dbh_df.iterrows():
        tree_id = int(row["Tree"])
        if tree_id in tree_locations_dict:
            xyz = tree_locations_dict[tree_id]
            dbh = row["DBH [cm]"]
            tree_dict[tree_id] = (xyz, dbh)
        else:
            logger.info(
                "for plot",
                plot_id,
                "could not find tree_id",
                tree_id,
                "in the tree_detections although dbh information exists",
            )
    return tree_dict


def test_dbh_arrays(inventory_csv, plot_id, tree_locations_csv):
    tree_dict = get_tree_id_to_xyz_dbh(inventory_csv, plot_id, tree_locations_csv)
    target_xyz_array = np.array([data[0] for data in tree_dict.values()])
    target_dbh_array = np.array([data[1] / 100 for data in tree_dict.values()])
    assert (
        target_xyz_array.ndim == 2
    ), f"target xyz is improper shape {tree_locations_csv}"
    assert (
        target_dbh_array.ndim == 1
    ), f"target dbh array is improper shape {tree_locations_csv}"


def evaluate_cylinders(
    cylinders: list[Cylinder],
    inventory_csv: Path,
    plot_id: str,
    tree_locations_csv: Path,
):
    tree_dict = get_tree_id_to_xyz_dbh(inventory_csv, plot_id, tree_locations_csv)
    target_xyz_array = np.array([data[0] for data in tree_dict.values()])
    target_dbh_array = np.array([data[1] / 100 for data in tree_dict.values()])
    pred_xyz_array = np.array([cyl.center for cyl in cylinders])
    pred_dbh_array = np.array([2 * cyl.radius for cyl in cylinders])

    # the search for correspondences in xy plane, for each target_tree,
    # find the closest pred_tree within a radius bound
    kdtree = KDTree(pred_xyz_array[:, :2])
    dist, closest_pred_to_target = kdtree.query(
        target_xyz_array[:, :2], k=1, p=2, distance_upper_bound=0.7, workers=-1
    )

    valid_pred_to_target = closest_pred_to_target != pred_xyz_array.shape[0]
    associated_count = sum(valid_pred_to_target)
    recall = associated_count / len(target_dbh_array)
    logger.info(
        "associated",
        associated_count,
        "trees out of",
        len(target_dbh_array),
        "known trees, recall =",
        recall,
    )
    # loop through range(0, target_id_max) and diff target dbh and corresponding
    # closest instance pred dbh
    error = np.array(
        [
            target_dbh_array[idx] - pred_dbh_array[closest_pred_to_target[idx]]
            for idx in range(target_xyz_array.shape[0])
            if valid_pred_to_target[idx]
        ]
    )
    rmse = np.sqrt(np.mean(np.square(error)))
    print("===DBH Error===")
    print(f"RMSE: {rmse*100:.3f} cm")
    return recall, error, rmse


def one_round(
    inventory_csv: Path,
    plot_folder: Path | None,
    label_folder: Path | None,
    aggregated_cloud_pth: Path | None,
    visualize: bool = False,
):
    plot_id = extract_plot_id_from_path(plot_folder or aggregated_cloud_pth)
    if aggregated_cloud_pth is not None:
        cloud = Cloud.load(aggregated_cloud_pth)
        tree_locations_csv = aggregated_cloud_pth.with_name("tree_locations_in_map.csv")
    else:
        assert plot_folder is not None
        cloud = aggregate_cloud(plot_folder, label_folder)
        tree_locations_csv = plot_folder / "tree_locations_in_map.csv"
    test_dbh_arrays(inventory_csv, plot_id, tree_locations_csv)
    cloud = cloud.voxel_down_sample(__VOXEL_SIZE__)
    sem_conf_mask_for_ground = get_conf_mask(cloud, __GROUND_CONF_THRESHOLD__)
    ground_mask = (cloud.semantics == 1).ravel()
    ground_confident_mask = np.logical_and(ground_mask, sem_conf_mask_for_ground)
    ground_cloud = cloud.select_by_mask(ground_confident_mask)
    if visualize:
        render([ground_cloud])
    non_ground_cloud = cloud.select_by_mask(ground_confident_mask, invert=True)

    normalized_cloud = normalize_height(
        non_ground_cloud=non_ground_cloud, ground_cloud=ground_cloud
    )
    if visualize:
        render([normalized_cloud])
    stem_mask = (normalized_cloud.semantics == 3).ravel()
    sem_conf_mask_for_stem = get_conf_mask(normalized_cloud, __STEM_CONF_THRESHOLD__)
    stem_confident_mask = np.logical_and(stem_mask, sem_conf_mask_for_stem)
    stem_cloud = normalized_cloud.select_by_mask(stem_confident_mask)
    if visualize:
        render([stem_cloud])
    clipped_cloud = clip_z(stem_cloud, min_z=0.5, max_z=4.0)
    clustered_cloud = project_2d_then_cluster_3d(clipped_cloud, k=300, beta=0.9)
    cylinders = fit_cylinders(
        clustered_cloud,
        ransac_iterations=10000,
        lsq_iterations=1000,
        voxel_size=__VOXEL_SIZE__,
    )
    recall, error, rmse = evaluate_cylinders(
        cylinders, inventory_csv, plot_id, tree_locations_csv
    )

    return plot_id, recall, error, rmse


def append_results_to_csv(run_number, run_name, rmse_map, avg_rmse, csv_filename):
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["run_name", "run_num"]
        fieldnames.extend(list(rmse_map.keys()))
        fieldnames.append("avg_rmse")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if file is empty, if so, write header
        if csvfile.tell() == 0:
            writer.writeheader()

        row = {"run_name": run_name, "run_num": run_number}
        for key, value in rmse_map.items():
            row[key] = value
        row["avg_rmse"] = avg_rmse
        writer.writerow(row)


def eval_loop(
    inventory_csv: Path,
    plot_folder: Path | None = None,
    label_folder: Path | None = None,
    aggregated_cloud_pth: Path | None = None,
    glob_dir: Path | None = None,
    visualize: bool = False,
):
    if glob_dir is not None:
        if plot_folder is not None or aggregated_cloud_pth is not None:
            print("glob_dir passed, ignoring everything else")

        plot_folders = sorted(
            [
                sub_dir
                for sub_dir in glob_dir.iterdir()
                if sub_dir.is_dir() and (sub_dir / "tree_locations_in_map.csv").exists()
            ]
        )
        # discard any extra inputs
        aggregated_cloud_pth = None
        label_folder = None
    else:
        plot_folders = [plot_folder]

    plot_rmse_map = {}
    for folder in plot_folders:
        print("working on", folder)
        plot_id, recall, error, rmse = one_round(
            inventory_csv, folder, label_folder, aggregated_cloud_pth, visualize
        )
        plot_rmse_map[plot_id] = {"rmse": rmse, "recall": recall}
    rmse_average = np.mean([val["rmse"] for val in plot_rmse_map.values()])
    print(plot_rmse_map)
    print("average rmse = ", rmse_average)
    return plot_rmse_map, rmse_average


def main(
    exp_name: str = typer.Argument(
        ..., help="Name of the experiment for tracking results."
    ),
    inventory_csv: Path = typer.Argument(
        ..., help="Path to the inventory CSV file containing reference data."
    ),
    plot_folder: Path | None = typer.Option(
        None,
        help="Optional path to the plot folder containing the point clouds and labels.",
    ),
    label_folder: Path | None = typer.Option(
        None, help="Optional path to the folder containing labels."
    ),
    aggregated_cloud_pth: Path | None = typer.Option(
        None, help="Optional path to the aggregated point cloud file."
    ),
    glob_dir: Path | None = typer.Option(
        None,
        help="Optional path to a directory for glob-based file loading. Will run the approach on each plot folder inside this folder",
    ),
    visualize: bool = typer.Option(
        False, help="Enable visualization of results if set to True."
    ),
):
    """
    Run cylinder fitting evaluation on DigiForests dataset plots and log results.

    This function performs the evaluation process for tree stem detection and
    diameter measurement using cylinder fitting. It can process a single plot or
    multiple plots.

    Args:
        exp_name: Unique identifier for the experiment run.
        inventory_csv: Path to the CSV file containing ground truth inventory data.
        plot_folder: Path to a single plot's data (mutually exclusive with glob_dir).
        label_folder: Path to semantic labels if not in plot_folder.
        aggregated_cloud_pth: Path to a pre-aggregated point cloud file.
        glob_dir: Directory containing multiple plot folders for batch processing.
        visualize: If True, enables visualization of intermediate results.

    Workflow:
    1. Runs evaluation loop on specified plot(s)
    2. Calculates per-plot and average RMSE for diameter estimation
    3. Appends results to a CSV file for experiment tracking

    Output:
    - Writes results to 'cylinder_fitting_results.csv' in the current directory.
    - CSV columns: run_name, run_num, [plot_ids], avg_rmse

    Note:
    - If glob_dir is provided, it takes precedence over plot_folder and aggregated_cloud_pth.
    - Visualization may significantly increase runtime and is intended for debugging.
    """
    filename = "cylinder_fitting_results.csv"
    plot_rmse_map, rmse_average = eval_loop(
        inventory_csv,
        plot_folder,
        label_folder,
        aggregated_cloud_pth,
        glob_dir,
        visualize,
    )
    append_results_to_csv(iter, exp_name, plot_rmse_map, rmse_average, filename)


if __name__ == "__main__":
    typer.run(main)

import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os

def transformation_to_rpy_tvec(T):
    """
    Convert a 4x4 transformation matrix to roll-pitch-yaw (rpy) and translation vector.

    Parameters:
        T (np.ndarray): 4x4 transformation matrix.

    Returns:
        rpy (3x1 np.array): [roll, pitch, yaw] in radians.
        tvec (3x1 np.array): translation vector.
    """
    if T.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")

    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rpy = Rotation.from_matrix(R).as_euler('xyz').reshape(3, 1)
    return rpy, tvec


def quat_to_rvec_tvec(pose):
    """
    Convert [x, y, z, qx, qy, qz, qw] to OpenCV rvec and tvec.

    Parameters:
        pose (list or np.array): 7 elements [x, y, z, qx, qy, qz, qw]

    Returns:
        rvec (3x1 np.array), tvec (3x1 np.array)
    """

    #Extract pose:
    x, y, z, qx, qy, qz, qw = pose

    #Convert to translation and rotation vectors:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    rvec = Rotation.from_quat(q).as_rotvec().reshape(3, 1)
    tvec = np.array([[x], [y], [z]])

    return rvec, tvec


def quat_to_transformation(pose):
    """
    Convert [x, y, z, qx, qy, qz, qw] to OpenCV rvec and tvec.

    Parameters:
        pose (list or np.array): 7 elements [x, y, z, qx, qy, qz, qw]

    Returns:
        rvec (3x1 np.array), tvec (3x1 np.array)
    """
    #Extract pose:
    x, y, z, qx, qy, qz, qw = pose

    #Convert to translation and rotation vectors:
    r = Rotation.from_quat([qx, qy, qz, qw])
    R_matrix = r.as_matrix()

    T = np.eye(4)
    T[0:3, 0:3] = R_matrix
    T[0:3, 3] = np.array([x, y, z])

    return T


def transform_points_tf_matrix(points, T):
    """
    Transform a set of 3D points using a 4x4 transformation matrix.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points.
        T (np.ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Nx3 array of transformed 3D points.
    """
    # Check shape
    if points.shape[1] != 3 or T.shape != (4, 4):
        raise ValueError("Expected points of shape (N, 3) and T of shape (4, 4)")

    # Convert points to homogeneous coordinates: (N, 4)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply transformation: (N, 4)
    transformed_h = points_h @ T.T

    # Convert back to 3D coordinates
    transformed_points = transformed_h[:, :3] / transformed_h[:, 3][:, np.newaxis]

    return transformed_points

def transform_points(points, rvec, tvec):
    """
    Transform 3D points by rvec and tvec.

    Args:
        points: Nx3 array of 3D points
        rvec: (3,1) rotation vector
        tvec: (3,1) translation vector

    Returns:
        transformed_points: Nx3 array of transformed points
    """
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)  # R is 3x3

    # Apply transformation
    points_transformed = (R @ points.T).T + tvec.T  # shape: Nx3

    return points_transformed


def parse_timestamp(filename, prefix):
    """Extract the timestamp (seconds and nanoseconds) from a filename."""
    parts = filename.replace(prefix, "").replace(".pcd", "").replace(".jpg", "").split("_")
    return int(parts[0]), int(parts[1])

def find_closest_images(pcd_folder, image_folder):
    """Find the closest image file for each point cloud file."""
    pcd_files = [f for f in os.listdir(pcd_folder) if f.endswith(".pcd")]
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    # Parse timestamps
    pcd_timestamps = [(f, *parse_timestamp(f, "cloud_")) for f in pcd_files]
    image_timestamps = [(f, *parse_timestamp(f, "image_")) for f in image_files]

    closest_matches = []

    for pcd_file, pcd_sec, pcd_nsec in pcd_timestamps:
        pcd_time = pcd_sec + pcd_nsec * 1e-9
        min_diff = float("inf")
        closest_image = None

        for image_file, img_sec, img_nsec in image_timestamps:
            img_time = img_sec + img_nsec * 1e-9
            time_diff = abs(pcd_time - img_time)

            if time_diff < min_diff:
                min_diff = time_diff
                closest_image = image_file

        closest_matches.append((pcd_file, closest_image))

    # Sort matches by LiDAR file name (or timestamp)
    closest_matches.sort(key=lambda x: x[0])
    return closest_matches


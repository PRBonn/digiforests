# pip install rosbags rosbags-image opencv-python typer tqdm
from pathlib import Path

import cv2
import numpy as np
import typer
from rosbags.image import message_to_cvimage
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

typestore = get_typestore(Stores.ROS1_NOETIC)
app = typer.Typer(add_completion=False)

IMAGE_TOPICS = [
    "/alphasense_driver_ros/cam0/debayered/image/compressed",
    "/alphasense_driver_ros/cam1/debayered/image/compressed",
    "/alphasense_driver_ros/cam2/debayered/image/compressed",
]

# "png" or "jpg"
IMAGE_FORMAT = "png"

limit = 500


def ensure_out_dir(base_out_dir: Path, topic: str) -> Path:
    """Return output directory for a given topic, creating it if needed."""
    parts = topic.split("/")
    cam_name = next((p for p in parts if p.startswith("cam")), "cam")
    cam_dir = base_out_dir / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    return cam_dir


def save_image(img: np.ndarray, out_dir: Path, sec: int, nsec: int) -> None:
    """Save image to out_dir/image_sec_nsec.<ext>."""
    fname = f"image_{sec}_{nsec}.{IMAGE_FORMAT}"
    out_path = out_dir / fname
    if IMAGE_FORMAT.lower() == "png":
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])


@app.command()
def extract(
    bag_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        "-o",
        exists=False,
        file_okay=False,
        writable=True,
        help="Output directory (cam0/, cam1/, cam2/ created inside)",
    ),
) -> None:
    """Extract images from all ROS1 .bag files in bag_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    bag_paths = sorted(p for p in bag_dir.iterdir() if p.suffix == ".bag")
    total_bags = len(bag_paths)

    for i, bag_path in enumerate(bag_paths, 1):
        typer.echo(f"Processing bag {i}/{total_bags}: {bag_path.name}")
        with Reader(bag_path) as reader:
            for conn in reader.connections:
                if conn.topic not in IMAGE_TOPICS:
                    continue

                cam_dir = ensure_out_dir(out_dir, conn.topic)
                msg_count = reader.topics[conn.topic].msgcount

                count = 0
                for connection, timestamp, rawdata in tqdm(
                    reader.messages(connections=[conn]),
                    total=msg_count,
                    desc=conn.topic,
                    unit="img",
                ):
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    cv_img = message_to_cvimage(msg)
                    sec = int(msg.header.stamp.sec)
                    nsec = int(msg.header.stamp.nanosec)
                    save_image(cv_img, cam_dir, sec, nsec)
                    count += 1
                    if count > limit:
                        break


if __name__ == "__main__":
    app()

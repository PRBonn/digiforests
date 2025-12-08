Make a folder where you plan to extract data from a rosbag, for example,

```bash
mkdir 2023_04_exp06_m3
```

make a dir for images specifically

```bash
mkdir 2023_04_exp06_m3/images
```

extract raw images from the rosbags

```bash
pip install rosbags rosbags-image opencv-python typer tqdm
cd scripts/data
python extract_images.py <path_to_dataset>/raw/train/2023-03/exp06-m3/rosbags/ -o <path_to_output_dir>/2023_04_exp06_m3/images
```

the images should get dumped in the following fashion

```bash
2023_04_exp06_m3/
└── images
    ├── cam0 # will have a bunch of time stamped png files
    ├── cam1 # will have a bunch of time stamped png files
    └── cam2 # will have a bunch of time stamped png files
```

change the cam folder names as appropriate based on the extrinsics yaml for the appropriate sequence date. For example, for the above sequence it's `scripts/calibration/extrinsics/sensors_frn015-mar23.yaml`. There `cam0 -> cam_front, cam1 -> cam_left, cam2 -> cam_right`. This rename is necessary to match the lidar point overlay example notebook in the calibration scripts folder.

Then, to extract the lidar scans, which are also motion deskewed, install

```bash
pip install rko_lio rosbags open3d
```

and then

```bash
rko_lio <path_to_dataset>/raw/train/2023-03/exp06-m3/rosbags/ \
  --imu_frame alphasense_imu_T_BI \
  --lidar_frame hesai_lidar_T_BL \
  --log \
  --log_dir <path_to_output_dir>/2023_04_exp06_m3 \
  --run_name odometry_output \
  --dump_deskewed 
```

That will result in a folder dump similar to

```bash
2023_04_exp06_m3/
├── images
│   ├── cam_front
│   ├── cam_left
│   └── cam_right
└── odometry_output_0
    ├── deskewed_scans
    └── odometry_output_tum_0.txt
```

`mv` the deskewed_scans folder outsider, and rename the `odometry_output_tum_0.txt` file to `poses.txt` (its a poses file in TUM format). You'll need to rename the deskewed scans to align with what is expected in the calibration example notebook in `scripts/calibration/image_lidar_overlay.ipynb`.

```python
from pathlib import Path

def main():
    scan_dir = Path("<path_to_output_dir>/2023_04_exp06_m3/deskewed_scans")
    for ply in scan_dir.glob("*.ply"):
        ts = int(ply.stem)
        sec = ts // 10**9
        nsec = ts % 10**9
        new_name = f"cloud_{sec}_{nsec:09d}.ply"
        new_path = ply.with_name(new_name)
        ply.rename(new_path)
        print(f"Renamed {ply.name} -> {new_name}")

if __name__ == "__main__":
    main()
```

The above is a simple rename script. Run that on the appropriate directory (change `scan_dir`).
You should finally end up with something that looks like

```bash
2023_04_exp06_m3/
├── deskewed_scans
│   ├── cloud_1679392256_249250048.ply
│   ├── cloud_1679392256_349269760.ply
│   ├── cloud_1679392256_449117952.ply
│   ...
├── images
│   ├── cam_front
│   │   ├── image_1679392256_110413870.png
│   │   ├── image_1679392256_160412970.png
│   │   ├── image_1679392256_210411970.png
│       ...
│   ├── cam_left
│   │   ├── image_1679392256_110413870.png
│   │   ├── image_1679392256_160412970.png
│   │   ├── image_1679392256_210411970.png
│       ...
│   └── cam_right
│       ├── image_1679392256_110413870.png
│       ├── image_1679392256_160412970.png
│       ├── image_1679392256_210411970.png
│       ...
└── lidar_odometry_poses_tum.txt
```

In `image_lidar_overlay.ipynb` you can now change the variables 

```python
sensor_yaml = "extrinsics/sensors_frn015-mar23.yaml"
slam_folder = "<path_to_output_dir>/2023_04_exp06_m3/"
pcd_folder = os.path.join(slam_folder, "deskewed_scans")
image_folder = os.path.join(slam_folder , "images")
cam_folder = os.path.join(image_folder , "cam_left")
```

to all point to the appropriate locations as shown above, and the point overlay on the images should work. Note that the lidar scans extracted by `rko_lio` are in lidar frame, and not base frame. This has to be accounted for when running the notebook.
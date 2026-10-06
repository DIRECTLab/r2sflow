#!/usr/bin/env python3
"""Convert simulated navigation episodes (ROS 2 bags) into KITTI-style scans for R2Flow.

Each episode directory holds `episode.json` and a ROS 2 sqlite bag under `bag/`.
Unlike the Livox bags `rosbag_to_r2flow.py` handles, one `/registered_scan`
message here is already a full 360 deg scan -- but it is registered into the
`map` frame. `/tf` carries `map -> sensor_at_scan` with exactly the scan's stamp,
so each scan is moved back into the sensor frame with the inverse of that pose
and written as

    float32 [x, y, z, reflectance] * N,  reflectance = 0 (the topic has no intensity)

The points are what `r2flow/data/go2w_nav_sim/go2w_nav_sim.py` spherically
projects. Measured in the sensor frame the scans span about -24..+11 deg of
elevation, so the default grid is 64 x 1024 over [-24, +12]: the KITTI-360
resolution, which keeps the pretrained HDiT's positional embedding loadable.

Train/test is split by episode so the two never share a stretch of trajectory.

Run this in a ROS 2 Humble environment (it needs `rosbag2_py`, `rclpy` and
`sensor_msgs_py`), not the R2Flow one:

    conda run -n ros2_humble_env python tools/ros2bag_registered_to_r2flow.py \
        --episodes-dir r2flow/data/simulated_bags \
        --out /path/with/space/go2w_nav_sim_r2flow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidar_grid import CH_MASK, RayGrid, save_scan, write_dataset_metadata

# Keep in sync with r2flow.utils.lidar.get_go2w_nav_linear_ray_angles; the
# dataset builder asserts on these.
H_DEFAULT = 64
W_DEFAULT = 1024
H_UP = 12.0
H_DOWN = -24.0


def _stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def iter_sensor_scans(bag_dir: Path, args):
    """Yield (stamp_ns, xyz in the sensor frame) for every scan with a matching pose.

    A scan and its `map -> sensor_at_scan` tf share a stamp but may be recorded in
    either order, so whichever arrives first waits for the other.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic, "/tf"]))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pending: dict[int, np.ndarray] = {}

    def to_sensor(xyz, pose):
        rotation, translation = pose
        return (xyz - translation) @ rotation  # R^T (p - t), row-vector form

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        msg = deserialize_message(raw, types[topic])
        if topic == "/tf":
            for tf in msg.transforms:
                if tf.header.frame_id != args.map_frame or tf.child_frame_id != args.sensor_frame:
                    continue
                stamp = _stamp_ns(tf.header.stamp)
                t = tf.transform.translation
                pose = (_quat_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z]))
                if stamp in pending:
                    yield stamp, to_sensor(pending.pop(stamp), pose)
                else:
                    poses[stamp] = pose
        else:
            stamp = _stamp_ns(msg.header.stamp)
            xyz = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
            xyz = xyz.astype(np.float64)
            if stamp in poses:
                yield stamp, to_sensor(xyz, poses.pop(stamp))
            else:
                pending[stamp] = xyz

    if pending:
        print(f"  {len(pending)} scans had no '{args.map_frame} -> {args.sensor_frame}' pose, skipped")


def convert(args) -> None:
    episodes = sorted(p for p in args.episodes_dir.iterdir() if (p / "bag").is_dir())
    if not episodes:
        raise SystemExit(f"no <episode>/bag directories in {args.episodes_dir}")
    n_test = args.test_episodes
    if not 0 < n_test < len(episodes):
        raise SystemExit(f"--test-episodes must be in [1, {len(episodes) - 1}]")
    test_names = {p.name for p in episodes[-n_test:]}

    grid = RayGrid(height=args.height, width=args.width, h_up=args.h_up, h_down=args.h_down)
    out_dir = args.out
    bin_dir = out_dir / "velodyne_points" / "data"
    bin_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for episode in episodes:
        split = "test" if episode.name in test_names else "train"
        print(f"{episode.name} ({split})")
        n_before = len(frames)
        for stamp, xyz in iter_sensor_scans(episode / "bag", args):
            depth = np.linalg.norm(xyz, axis=1)
            xyz = xyz[(depth >= args.min_depth) & (depth <= args.max_depth)]
            reflectance = np.zeros(len(xyz))
            sample_id = len(frames)
            file_name = f"{sample_id:010d}.bin"
            save_scan(bin_dir / file_name, xyz, reflectance)
            image = grid.project(xyz, reflectance, args.min_depth, args.max_depth)
            frames.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "episode": episode.name,
                    "stamp": stamp * 1e-9,
                    "num_points": int(len(xyz)),
                    "pixel_fill": float(image[..., CH_MASK].mean()),
                    "split": split,
                }
            )
        print(f"  {len(frames) - n_before} scans")

    if not frames:
        raise SystemExit("no scans written")

    write_dataset_metadata(
        out_dir,
        frames,
        {
            "source": "simulated navigation episodes (ROS 2 bags)",
            "topic": args.topic,
            "frame_id": args.sensor_frame,
            "num_rings": grid.height,
            "image_width": grid.width,
            "h_up_deg": grid.h_up,
            "h_down_deg": grid.h_down,
            "elevation_top_deg": grid.h_up,
            "elevation_step_deg": grid.elevation_step,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "intensity_scale": None,
            "num_frames": len(frames),
            "num_train": sum(f["split"] == "train" for f in frames),
            "num_test": sum(f["split"] == "test" for f in frames),
            "test_episodes": sorted(test_names),
        },
    )

    fill = np.array([f["pixel_fill"] for f in frames])
    print(f"wrote {len(frames)} scans to {bin_dir}")
    print(f"  split:       {sum(f['split'] == 'train' for f in frames)} train / "
          f"{sum(f['split'] == 'test' for f in frames)} test")
    print(f"  pixel fill:  mean={fill.mean():.3f} min={fill.min():.3f} (of {grid.height}x{grid.width})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("input")
    src.add_argument("--episodes-dir", type=Path, default=Path("r2flow/data/simulated_bags"),
                     help="directory of <episode>/bag ROS 2 bags")
    src.add_argument("--topic", default="/registered_scan")
    src.add_argument("--map-frame", default="map")
    src.add_argument("--sensor-frame", default="sensor_at_scan")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("r2flow/data/go2w_nav_sim/dataset"))
    out.add_argument("--height", type=int, default=H_DEFAULT)
    out.add_argument("--width", type=int, default=W_DEFAULT)
    out.add_argument("--h-up", type=float, default=H_UP,
                     help="elevation of the top row, degrees (default: 12.0)")
    out.add_argument("--h-down", type=float, default=H_DOWN,
                     help="elevation one step below the bottom row (default: -24.0)")
    out.add_argument("--min-depth", type=float, default=0.5)
    out.add_argument("--max-depth", type=float, default=30.0)
    out.add_argument("--test-episodes", type=int, default=2,
                     help="the last N episodes (sorted by name) form the test split")

    convert(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Convert a real Unitree Go2 (utlidar L1) ROS 2 bag into R2Flow range images.

The L1 is a spinning non-repetitive scanner: one PointCloud2 message is a ~65 ms
slice of a continuous spiral, not a full sweep, and its `ring` field is a
constant so there is no row structure to exploit. This tool accumulates several
consecutive messages into one sweep and projects them onto the same ray grid the
go2w_sim checkpoints were trained on, so the result can be fed straight to the
model.

Unlike the sim converter there is no ego-motion de-skewing: this bag carries no
/tf and no odometry for the lidar frame, so accumulated sweeps smear by whatever
the robot did during the window. Keep --messages-per-scan small.

Run in the light bag venv (numpy + rosbags), not the torch container:

    .venv-rosbag/bin/python tools/rosbag2_go2_to_r2flow.py \
        --bag /mnt/fast/lidar_data/go2_bags/go2_mocam_loop \
        --out /mnt/fast/lidar_data/go2_real_r2flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bag_io
import lidar_grid
from lidar_grid import RayGrid


def accumulate_sweeps(
    bag: Path, topic: str, per_scan: int, typestore: str, limit: int | None
):
    """Group consecutive messages into sweeps of `per_scan` messages each."""
    buf_xyz, buf_ref, buf_time, buf_stamp = [], [], [], []
    emitted = 0
    for stamp, points in bag_io.iter_pointclouds([bag], topic, typestore):
        xyz, intensity = bag_io.as_xyzi(points)
        if len(xyz) == 0:
            continue
        buf_xyz.append(xyz)
        buf_ref.append(intensity)
        sweep_time = bag_io.field_or_none(points, "time")
        buf_time.append(
            sweep_time[: len(xyz)] if sweep_time is not None else np.zeros(len(xyz))
        )
        buf_stamp.append(stamp)
        if len(buf_xyz) < per_scan:
            continue
        yield (
            np.concatenate(buf_xyz),
            np.concatenate(buf_ref),
            np.concatenate(buf_time),
            float(np.median(buf_stamp)),
            float(buf_stamp[-1] - buf_stamp[0]),
        )
        buf_xyz, buf_ref, buf_time, buf_stamp = [], [], [], []
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def check_origin(args) -> None:
    """Report whether the cloud is sensor-centred, using ONE message.

    Must not run on an accumulated sweep: the per-point `time` field restarts at
    zero every message, so concatenating them interleaves the sweep order and
    the elevation trace tears regardless of which origin is correct.
    """
    for _, points in bag_io.iter_pointclouds([args.bag], args.topic, args.typestore):
        xyz, _ = bag_io.as_xyzi(points)
        sweep_time = bag_io.field_or_none(points, "time")
        if sweep_time is not None:
            sweep_time = sweep_time[: len(xyz)]
        report = lidar_grid.diagnose_origin(xyz, sweep_time)
        break
    else:
        return

    print("\norigin check on a single message"
          " (want: large void_radius, small elevation_step_p99)")
    for name, m in report.items():
        print(
            f"  {name:14s} void_radius={m['void_radius']:.3f} m"
            f"  elevation_step_p99={m.get('elevation_step_p99', float('nan')):.2f} deg"
        )
    best = min(report, key=lambda k: report[k].get("elevation_step_p99", 1e9))
    origin = tuple(round(v, 3) for v in args.sensor_origin)
    print(f"  smoothest sweep from {best}; projecting from {origin}")
    if np.allclose(args.sensor_origin, 0) and best != "(0,0,0)":
        print("  WARNING: the data prefers a different origin. "
              "Re-run with --sensor-origin X Y Z if the images look wrong.")
    print()


def convert(args) -> None:
    grid = RayGrid(args.height, args.width, args.h_up, args.h_down)
    out_dir = Path(args.out)
    bin_dir = out_dir / "velodyne_points" / "data"
    bin_dir.mkdir(parents=True, exist_ok=True)

    print(f"bag:    {args.bag}")
    for topic, msgtype, count in bag_io.list_topics([args.bag], args.typestore):
        mark = " <-" if topic == args.topic else ""
        print(f"  {topic:26s} {msgtype:32s} {count}{mark}")
    print(
        f"grid:   {grid.height}x{grid.width}  elevation "
        f"[{grid.h_down:.2f}, {grid.h_up:.2f}] deg, step {grid.elevation_step:.3f}"
    )

    check_origin(args)

    origin = np.array(args.sensor_origin, dtype=np.float64)
    frames, fills = [], []

    for xyz, intensity, _sweep_time, stamp, span in accumulate_sweeps(
        args.bag, args.topic, args.messages_per_scan, args.typestore, args.limit
    ):
        if args.flip_z:
            xyz = xyz * np.array([1.0, 1.0, -1.0])
        image = grid.project(
            xyz - origin,
            np.clip(intensity / args.intensity_scale, 0.0, 1.0),
            args.min_depth,
            args.max_depth,
        )
        mask = image[..., lidar_grid.CH_MASK]
        keep = mask > 0
        lidar_grid.save_scan(
            bin_dir / f"{len(frames):010d}.bin",
            image[keep][:, [lidar_grid.CH_X, lidar_grid.CH_Y, lidar_grid.CH_Z]],
            image[keep][:, lidar_grid.CH_REFLECTANCE],
        )
        fills.append(float(mask.mean()))
        frames.append(
            {
                "sample_id": len(frames),
                "file_name": f"{len(frames):010d}.bin",
                "t_ref": stamp,
                "num_messages": args.messages_per_scan,
                "window_seconds": span,
                "num_points": int(keep.sum()),
                "pixel_fill": fills[-1],
            }
        )

    if not frames:
        raise SystemExit("no sweeps produced -- check --topic")

    n_test = max(1, int(round(len(frames) * args.test_fraction)))
    for f in frames:
        f["split"] = "test" if f["sample_id"] >= len(frames) - n_test else "train"

    lidar_grid.write_dataset_metadata(
        out_dir,
        frames,
        {
            "source": "unitree go2 (utlidar L1) ros2 bag",
            "bag": str(args.bag),
            "topic": args.topic,
            "num_rings": grid.height,
            "image_width": grid.width,
            "h_up_deg": grid.h_up,
            "h_down_deg": grid.h_down,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "intensity_scale": args.intensity_scale,
            "sensor_origin": list(map(float, origin)),
            "ego_motion_compensated": False,
            "messages_per_scan": args.messages_per_scan,
            "num_frames": len(frames),
        },
    )

    fills = np.array(fills)
    spans = np.array([f["window_seconds"] for f in frames])
    pts = np.array([f["num_points"] for f in frames])
    print(f"wrote {len(frames)} sweeps to {bin_dir}")
    print(f"  window:     mean={spans.mean():.3f}s max={spans.max():.3f}s")
    print(f"  points/scan mean={pts.mean():.0f}")
    print(f"  pixel fill: mean={fills.mean():.4f} min={fills.min():.4f} max={fills.max():.4f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bag", type=Path, required=True, help="rosbag2 directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--topic", default="/go2/raw_lidar")
    p.add_argument("--typestore", default="ros2_humble",
                   choices=("ros2_humble", "ros2_jazzy", "ros1_noetic"))
    p.add_argument("--messages-per-scan", type=int, default=8,
                   help="consecutive messages merged into one sweep (default: 8)")
    p.add_argument("--limit", type=int, default=None, help="stop after N sweeps")
    # Defaults describe the L1 grid (lower hemisphere), matching
    # `rosbag_to_r2flow.py --sensor l1` and r2flow/data/go2w_sim_l1. The
    # elevation band must be given explicitly alongside height/width -- setting
    # only --height silently resamples whatever band is in force.
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--h-up", type=float, default=0.0,
                   help="elevation of the top row, degrees (default: 0)")
    p.add_argument("--h-down", type=float, default=-90.0,
                   help="elevation one step below the bottom row (default: -90)")
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=30.0)
    p.add_argument("--intensity-scale", type=float, default=255.0,
                   help="divisor mapping raw intensity to [0,1] (default: 255)")
    p.add_argument("--sensor-origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="project rays from here instead of the frame origin")
    # The utlidar frame is z-down: every return has positive z and the dense
    # plane at z=0.347 is the floor under a sensor ~0.35 m up. The sim is z-up,
    # so flip to put both in the same convention (real data then lands in the
    # lower hemisphere, which is what the L1 grid covers).
    p.add_argument("--flip-z", action="store_true", default=True,
                   help="negate z to convert the z-down utlidar frame (default: on)")
    p.add_argument("--no-flip-z", dest="flip_z", action="store_false",
                   help="keep z as published")
    p.add_argument("--test-fraction", type=float, default=0.2)
    convert(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

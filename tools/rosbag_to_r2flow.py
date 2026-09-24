#!/usr/bin/env python3
"""Convert Unitree Go2-W simulator rosbags into KITTI-style LiDAR scans for R2Flow.

The simulated Livox on `/<robot>/livox/lidar` publishes one ~60 deg azimuth sector
per message, so a single message is not a full scan. This script accumulates
consecutive messages until the full 360 deg is covered, de-skews each sector into
the sensor pose at the window's reference time using `odom -> base` from the tf
bag, and writes one KITTI-style `.bin` per assembled scan:

    float32 [x, y, z, reflectance] * N,  reflectance in [0, 1]

The points stay in the `livox_frame` of the reference timestamp, which is what
`r2flow/data/go2w_sim/go2w_sim.py` expects when it spherically projects them onto
the sensor's native 40 x 500 ray grid.

Run this outside the R2Flow environment -- it only needs `numpy` and `rosbags`
(`uv pip install rosbags numpy`), not torch.

Example:

    python tools/rosbag_to_r2flow.py \
        --bag-dir /mnt/fast/lidar_data/rosbag_20260917_204734_246468 \
        --out r2flow/data/go2w_sim/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bag_io
from bag_io import decode_pointcloud
from bag_io import stamp_seconds as _stamp_seconds
from lidar_grid import RayGrid

# =====================================================================================
# Sensor geometry
# =====================================================================================
# Measured from the bags and exact to ~1e-5 deg:
#   elevation(ring) = 29.50 - 1.48 * ring,  ring in [0, 39]
#   azimuth lands on a 0.72 deg grid       -> 500 columns for a full turn
#
# Written in the linear ray-angle form the repo already uses so that
# r2flow.utils.lidar.get_go2w_livox_linear_ray_angles reproduces the table exactly:
#   elevation(row) = (1 - row / H) * (H_UP - H_DOWN) + H_DOWN     (row 0 = topmost)
# Keep these in sync with that function; the dataset builder asserts on them.
H_DEFAULT = 40
W_DEFAULT = 500
H_UP = 29.50
H_DOWN = -29.70


# =====================================================================================
# tf handling
# =====================================================================================
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternions -> (..., 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between (M, 4) xyzw quaternions."""
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0)[:, None], -q1, q1)  # take the shorter arc
    theta = np.arccos(np.abs(dot).clip(-1.0, 1.0))
    sin_theta = np.sin(theta)
    tiny = theta < 1e-6  # fall back to lerp when the arc is degenerate
    denom = np.where(tiny, 1.0, sin_theta)
    s0 = np.where(tiny, 1.0 - u, np.sin((1.0 - u) * theta) / denom)
    s1 = np.where(tiny, u, np.sin(u * theta) / denom)
    q = s0[:, None] * q0 + s1[:, None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


@dataclass
class PoseTrack:
    """Time-indexed rigid transforms, linearly interpolated (slerp for rotation)."""

    time: np.ndarray  # (N,) seconds, sorted
    trans: np.ndarray  # (N, 3)
    quat: np.ndarray  # (N, 4) xyzw

    def matrices_at(self, times: np.ndarray) -> np.ndarray:
        """(M,) timestamps -> (M, 4, 4) homogeneous transforms."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        hi = np.searchsorted(self.time, times).clip(1, len(self.time) - 1)
        lo = hi - 1
        span = self.time[hi] - self.time[lo]
        u = np.where(span > 0, (times - self.time[lo]) / np.where(span > 0, span, 1.0), 0.0)
        u = u.clip(0.0, 1.0)  # clamp instead of extrapolating past the track ends
        out = np.zeros((len(times), 4, 4), dtype=np.float64)
        out[:, :3, :3] = _quat_to_matrix(_slerp(self.quat[lo], self.quat[hi], u))
        out[:, :3, 3] = self.trans[lo] + u[:, None] * (self.trans[hi] - self.trans[lo])
        out[:, 3, 3] = 1.0
        return out

    def covers(self, t: float) -> bool:
        return self.time[0] <= t <= self.time[-1]


def read_tf(bags: list[Path], parent: str, child: str) -> PoseTrack:
    """Collect every `/tf` sample of `parent -> child` into a PoseTrack."""
    time, trans, quat = [], [], []
    with bag_io.open_reader(bags, "ros1_noetic") as reader:
        conns = [c for c in reader.connections if c.topic == "/tf"]
        if not conns:
            raise SystemExit(f"no /tf topic in {[b.name for b in bags]}")
        for conn, _, raw in reader.messages(connections=conns):
            for tf in reader.deserialize(raw, conn.msgtype).transforms:
                if tf.header.frame_id != parent or tf.child_frame_id != child:
                    continue
                t, r = tf.transform.translation, tf.transform.rotation
                time.append(_stamp_seconds(tf.header))
                trans.append((t.x, t.y, t.z))
                quat.append((r.x, r.y, r.z, r.w))
    if not time:
        raise SystemExit(f"no '{parent} -> {child}' transform found on /tf")
    order = np.argsort(np.asarray(time))
    return PoseTrack(
        time=np.asarray(time, dtype=np.float64)[order],
        trans=np.asarray(trans, dtype=np.float64)[order],
        quat=np.asarray(quat, dtype=np.float64)[order],
    )


def read_tf_static(bags: list[Path], parent: str, child: str) -> np.ndarray:
    """Last-wins lookup of a `/tf_static` transform -> (4, 4)."""
    found = None
    with bag_io.open_reader(bags, "ros1_noetic") as reader:
        conns = [c for c in reader.connections if c.topic == "/tf_static"]
        for conn, _, raw in reader.messages(connections=conns):
            for tf in reader.deserialize(raw, conn.msgtype).transforms:
                if tf.header.frame_id == parent and tf.child_frame_id == child:
                    found = tf.transform
    if found is None:
        raise SystemExit(f"no '{parent} -> {child}' transform found on /tf_static")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_to_matrix(
        np.array([found.rotation.x, found.rotation.y, found.rotation.z, found.rotation.w])
    )
    mat[:3, 3] = (found.translation.x, found.translation.y, found.translation.z)
    return mat


# =====================================================================================
# Point handling
# =====================================================================================
def xyzi_of(points: np.ndarray) -> np.ndarray:
    """Structured array -> (N, 4) float64 [x, y, z, intensity], finite rows only."""
    xyz, intensity = bag_io.as_xyzi(points)
    return np.concatenate([xyz, intensity[:, None]], axis=1)


def grid_indices(
    xyz: np.ndarray, grid: RayGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-ray-cell indices; see lidar_grid.RayGrid for the convention.

    Used here only for coverage bookkeeping and fill statistics -- the dataset
    builder does the projection that actually reaches the model.
    """
    return grid.row_col(xyz)


# =====================================================================================
# Sensor presets
# =====================================================================================
# The Go2-W sim carries two lidars. They need different ray grids, so rather than
# make the caller remember six numbers each time, name them.
#
# `livox` is the back-mounted Livox (base -> livox_frame: x=+0.16, z=+0.14,
# pitch +13 deg): 40 uniform rings over a narrow band, ranges out to 70 m.
#
# `l1` is the front-mounted, downward L1 (base -> lidar: x=+0.29, z=-0.15,
# pitch -6.2 deg), the counterpart of the real robot's utlidar. Its `ring` field
# is a scan-order index, not an elevation row, so there is no ring table to use
# and the projection is purely spherical. It sweeps a full 360 deg vertical
# circle, but only the lower hemisphere is kept: the real L1 reports nothing
# above the horizon, so the upper half would be sim-only signal the model could
# never match. min_depth 0.1 keeps the dense ground returns directly beneath the
# robot while cutting most chassis self-hits; max_depth is the sim's own clamp.
SENSOR_PRESETS = {
    "livox": {
        "topic": "/go2w_sim_005/livox/lidar",
        "sensor_frame": "go2w_sim_005/livox_frame",
        "height": 40, "width": 500, "h_up": 29.50, "h_down": -29.70,
        "min_depth": 0.5, "max_depth": 80.0, "intensity_scale": 100.0,
    },
    "l1": {
        "topic": "/go2w_sim_005/lidar",
        "sensor_frame": "go2w_sim_005/lidar",
        "height": 64, "width": 512, "h_up": 0.0, "h_down": -90.0,
        "min_depth": 0.1, "max_depth": 30.0, "intensity_scale": 100.0,
        # Rays are 0.635 deg apart in azimuth against 0.703 deg cells, so even a
        # perfect rotation only reaches ~98% of the columns. Demanding more just
        # buys extra rotations: 0.97 costs 30% of the scans and a 44% longer
        # accumulation window to move fill from 0.797 to 0.834.
        "coverage": 0.90, "min_coverage": 0.85,
    },
}


def apply_sensor_preset(parser, args, argv) -> None:
    """Fill preset values, without clobbering anything given explicitly."""
    given = set(argv if argv is not None else sys.argv[1:])
    for key, value in SENSOR_PRESETS[args.sensor].items():
        flag = "--" + key.replace("_", "-")
        if flag not in given:
            setattr(args, key, value)
    print(f"sensor preset '{args.sensor}': topic={args.topic} "
          f"grid={args.height}x{args.width} "
          f"elevation[{args.h_down:.2f},{args.h_up:.2f}] "
          f"depth[{args.min_depth},{args.max_depth}]")


# =====================================================================================
# Frame assembly
# =====================================================================================
def assemble(args) -> None:
    sensor_bags = sorted(Path(args.bag_dir).glob(args.sensor_glob)) if args.bag_dir else list(map(Path, args.bags))
    if not sensor_bags:
        raise SystemExit(f"no sensor bags matched {args.sensor_glob!r} in {args.bag_dir}")
    tf_bags = sorted(Path(args.bag_dir).glob(args.tf_glob)) if args.bag_dir else list(map(Path, args.tf_bags or []))
    print(f"sensor bags: {[b.name for b in sensor_bags]}")

    odom_to_base = base_to_sensor = None
    if not args.no_compensate:
        if not tf_bags:
            where = f"matched {args.tf_glob!r} in {args.bag_dir}" if args.bag_dir else "given via --tf-bags"
            raise SystemExit(
                f"no tf bags {where}; de-skewing needs '{args.odom_frame} -> {args.base_frame}'. "
                "Pass --no-compensate to skip it."
            )
        print(f"tf bags:     {[b.name for b in tf_bags]}")
        odom_to_base = read_tf(tf_bags, args.odom_frame, args.base_frame)
        base_to_sensor = read_tf_static(tf_bags, args.base_frame, args.sensor_frame)
        print(
            f"poses:       {len(odom_to_base.time)} samples of "
            f"'{args.odom_frame} -> {args.base_frame}' "
            f"over {odom_to_base.time[-1] - odom_to_base.time[0]:.1f}s"
        )
        print(f"static:      '{args.base_frame} -> {args.sensor_frame}' t={base_to_sensor[:3, 3]}")

    out_dir = Path(args.out)
    bin_dir = out_dir / "velodyne_points" / "data"
    bin_dir.mkdir(parents=True, exist_ok=True)

    grid = RayGrid(args.height, args.width, args.h_up, args.h_down)
    H, W = grid.height, grid.width
    window: list[tuple[float, np.ndarray]] = []  # (timestamp, (N, 4) xyzi)
    covered = np.zeros(W, dtype=bool)
    frames: list[dict] = []
    n_msgs = n_dropped = n_uncovered = 0

    def flush() -> None:
        """Emit the buffered window as one scan, or discard it if too incomplete."""
        nonlocal window, covered, n_dropped
        if not window:
            return
        fill = float(covered.mean())
        if fill < args.min_coverage:
            n_dropped += 1
            window, covered = [], np.zeros(W, dtype=bool)
            return

        times = np.array([t for t, _ in window])
        t_ref = float(np.median(times)) if args.ref_time == "middle" else float(times[-1])

        if odom_to_base is None:
            merged = np.concatenate([p for _, p in window])
        else:
            # p_ref = inv(T_odom_sensor(t_ref)) @ T_odom_sensor(t_msg) @ p_msg
            mats = odom_to_base.matrices_at(times) @ base_to_sensor
            ref_inv = np.linalg.inv(odom_to_base.matrices_at([t_ref])[0] @ base_to_sensor)
            chunks = []
            for (_, pts), mat in zip(window, mats):
                rel = ref_inv @ mat
                xyz = pts[:, :3] @ rel[:3, :3].T + rel[:3, 3]
                chunks.append(np.concatenate([xyz, pts[:, 3:4]], axis=1))
            merged = np.concatenate(chunks)

        depth = np.linalg.norm(merged[:, :3], axis=1)
        keep = (depth > args.min_depth) & (depth < args.max_depth)
        merged, depth = merged[keep], depth[keep]

        row, col, valid = grid_indices(merged[:, :3], grid)
        occupied = np.zeros((H, W), dtype=bool)
        occupied[row[valid], col[valid]] = True

        scan = np.empty((len(merged), 4), dtype=np.float32)
        scan[:, :3] = merged[:, :3]
        scan[:, 3] = (merged[:, 3] / args.intensity_scale).clip(0.0, 1.0)
        scan.tofile(bin_dir / f"{len(frames):010d}.bin")

        frames.append(
            {
                "sample_id": len(frames),
                "file_name": f"{len(frames):010d}.bin",
                "t_ref": t_ref,
                "t_start": float(times[0]),
                "t_end": float(times[-1]),
                "num_messages": len(window),
                "num_points": int(len(merged)),
                "azimuth_coverage": fill,
                "pixel_fill": float(occupied.mean()),
            }
        )
        window, covered = [], np.zeros(W, dtype=bool)

    with bag_io.open_reader(sensor_bags, "ros1_noetic") as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {args.topic!r} not in bags. available: {topics}")
        total = sum(c.msgcount for c in conns)
        print(f"topic:       {args.topic} ({total} messages)\n")

        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = _stamp_seconds(msg.header)
            if odom_to_base is not None and not odom_to_base.covers(stamp):
                n_uncovered += 1
                continue
            pts = xyzi_of(decode_pointcloud(msg))
            if len(pts) == 0:
                continue
            n_msgs += 1

            _, col, valid = grid_indices(pts[:, :3], grid)
            covered[col[valid]] = True
            window.append((stamp, pts))

            if covered.mean() >= args.coverage:
                flush()
            elif stamp - window[0][0] > args.max_window:
                flush()  # stalled window: emit if usable, otherwise drop it

    flush()

    if not frames:
        raise SystemExit("no complete scans assembled -- try lowering --coverage")

    # Contiguous temporal split so train and test do not share a stretch of trajectory.
    n_test = max(1, int(round(len(frames) * args.test_fraction)))
    for frame in frames:
        frame["split"] = "test" if frame["sample_id"] >= len(frames) - n_test else "train"

    (out_dir / "frames.json").write_text(json.dumps(frames, indent=1))
    (out_dir / "sensor_info.json").write_text(
        json.dumps(
            {
                "source": "unitree go2w simulator rosbag",
                "topic": args.topic,
                "frame_id": args.sensor_frame,
                "num_rings": H,
                "native_azimuth_cells": W,
                "h_up_deg": grid.h_up,
                "h_down_deg": grid.h_down,
                "elevation_top_deg": grid.h_up,
                "elevation_step_deg": grid.elevation_step,
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "intensity_scale": args.intensity_scale,
                "ego_motion_compensated": odom_to_base is not None,
                "reference_time": args.ref_time,
                "num_frames": len(frames),
                "num_train": sum(f["split"] == "train" for f in frames),
                "num_test": sum(f["split"] == "test" for f in frames),
            },
            indent=1,
        )
    )

    fill = np.array([f["pixel_fill"] for f in frames])
    msgs = np.array([f["num_messages"] for f in frames])
    span = np.array([f["t_end"] - f["t_start"] for f in frames])
    print(f"wrote {len(frames)} scans to {bin_dir}")
    print(f"  split:       {sum(f['split'] == 'train' for f in frames)} train / {n_test} test")
    print(f"  msgs/scan:   mean={msgs.mean():.1f} max={msgs.max()}")
    print(f"  window:      mean={span.mean():.3f}s max={span.max():.3f}s")
    print(f"  pixel fill:  mean={fill.mean():.3f} min={fill.min():.3f} (of {H}x{W})")
    if n_dropped:
        print(f"  dropped {n_dropped} stalled windows below --min-coverage {args.min_coverage}")
    if n_uncovered:
        print(f"  skipped {n_uncovered} messages outside the tf time range")
    print(f"  consumed {n_msgs} of {total} messages")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("input")
    src.add_argument("--bag-dir", type=Path, help="directory holding the sensor and tf bags")
    src.add_argument("--bags", nargs="+", help="explicit sensor bag paths (instead of --bag-dir)")
    src.add_argument("--tf-bags", nargs="+", help="explicit tf bag paths (instead of --bag-dir)")
    src.add_argument("--sensor-glob", default="*_sensors_*.bag")
    src.add_argument("--tf-glob", default="*_tf_*.bag")
    src.add_argument("--topic", default="/go2w_sim_005/livox/lidar")
    src.add_argument("--odom-frame", default="go2w_sim_005/odom")
    src.add_argument("--base-frame", default="go2w_sim_005/base")
    src.add_argument("--sensor-frame", default="go2w_sim_005/livox_frame")

    asm = p.add_argument_group("frame assembly")
    asm.add_argument("--coverage", type=float, default=0.995,
                     help="azimuth-column fraction that completes a scan (default: 0.995)")
    asm.add_argument("--min-coverage", type=float, default=0.90,
                     help="keep a stalled window only above this coverage (default: 0.90)")
    asm.add_argument("--max-window", type=float, default=1.2,
                     help="seconds before a window is considered stalled (default: 1.2)")
    asm.add_argument("--ref-time", choices=("middle", "last"), default="middle",
                     help="de-skew target within the window (default: middle)")
    asm.add_argument("--no-compensate", action="store_true",
                     help="skip ego-motion de-skewing (no tf bag needed)")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("r2flow/data/go2w_sim/dataset"))
    # This grid decides when the azimuth counts as covered and drives the reported
    # fill; it is not the training resolution, which the dataset builder sets.
    out.add_argument("--height", type=int, default=H_DEFAULT,
                     help="rings in the sensor's native grid (default: 40)")
    out.add_argument("--width", type=int, default=W_DEFAULT,
                     help="azimuth cells used for coverage bookkeeping (default: 500)")
    out.add_argument("--h-up", type=float, default=H_UP,
                     help="elevation of the top row, degrees (default: 29.50)")
    out.add_argument("--h-down", type=float, default=H_DOWN,
                     help="elevation one step below the bottom row (default: -29.70)")
    out.add_argument("--min-depth", type=float, default=0.5)
    out.add_argument("--max-depth", type=float, default=80.0)
    out.add_argument("--intensity-scale", type=float, default=100.0,
                     help="divisor mapping raw intensity to [0, 1] (default: 100.0)")
    out.add_argument("--test-fraction", type=float, default=0.2)

    p.add_argument("--sensor", choices=("livox", "l1"), default=None,
                   help="preset for the two simulated sensors; sets --topic, "
                        "--sensor-frame, the ray grid and the depth/intensity ranges")

    args = p.parse_args(argv)
    if not args.bag_dir and not args.bags:
        p.error("pass either --bag-dir or --bags")
    if args.sensor:
        apply_sensor_preset(p, args, argv)
    assemble(args)


if __name__ == "__main__":
    sys.exit(main())

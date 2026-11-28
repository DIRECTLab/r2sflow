#!/usr/bin/env python3
"""Convert L2-emulating hospital-sim episodes (../lidargen, lidar_config=Unitree_L2)
into R2Flow scans, preprocessed exactly like the real Go2 sweeps.

Two kinds of episode directory are accepted. The first is `bag/` with
`/lidar_points` (collect_data.py). The second is `l2_scans.npz`
(replay_poses_l2.py), which holds recorded trajectories re-rendered with the L2.

The sim now records `/lidar_points`: raw sensor-frame scans of the emulated
utlidar, mounted with the real sensor's calibrated tilt, with no registration
and no motion compensation, the same as the real `/go2/raw_lidar`. So this
reuses `rosbag2_go2_to_r2flow.accumulate_sweeps` and the same z-flip and
`RayGrid` projection. The only differences are:

* message rate: the sim emits one full rotor turn (0.25 s) per message, while
  the real sensor emits 65 ms slices. `--messages-per-scan 2` therefore spans
  about the same 0.5 s window as the real 8 messages.
* split: whole episodes, the last `--test-episodes` held out, the same as
  `go2w_nav_sim`.

    python tools/sim_l2_to_r2flow.py --episodes-dir r2flow/data/rosbags/simulated_bags_l2 \\
        --out r2flow/data/go2w_l2_sim/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lidar_grid  # noqa: E402
from lidar_grid import RayGrid  # noqa: E402
from rosbag2_go2_to_r2flow import accumulate_sweeps  # noqa: E402


def episode_ok(ep: Path) -> bool:
    if (ep / "l2_scans.npz").exists():  # re-rendered trajectory (replay_poses_l2)
        return True
    meta = ep / "episode.json"
    if not (ep / "bag" / "metadata.yaml").exists() or not meta.exists():
        return False
    return not str(json.loads(meta.read_text()).get("exit_reason", "")).startswith("error")


def npz_sweeps(path: Path, per_scan: int, stride: int, limit: int | None):
    """accumulate_sweeps over lidargen's replay output: same yield signature
    (xyz, intensity, time, stamp, span)."""
    d = np.load(path)
    pts, off, t = d["points"], d["offsets"], d["t"]
    emitted = 0
    for end in range(per_scan, len(t) + 1, stride):
        start = end - per_scan
        xyz = pts[off[start]:off[end]].astype(np.float64)
        yield xyz, np.zeros(len(xyz)), np.zeros(len(xyz)), float(np.median(t[start:end])), float(t[end - 1] - t[start])
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-dir", type=Path, default=Path("r2flow/data/rosbags/simulated_bags_l2"))
    p.add_argument("--out", type=Path, default=Path("r2flow/data/go2w_l2_sim/dataset"))
    p.add_argument("--topic", default="/lidar_points")
    p.add_argument("--typestore", default="ros2_humble")
    p.add_argument("--messages-per-scan", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--h-up", type=float, default=0.0)
    p.add_argument("--h-down", type=float, default=-24.0)
    p.add_argument("--min-depth", type=float, default=0.5)
    p.add_argument("--max-depth", type=float, default=30.0)
    p.add_argument("--test-episodes", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="sweeps per episode (smoke tests)")
    args = p.parse_args(argv)

    grid = RayGrid(args.height, args.width, args.h_up, args.h_down)
    episodes = sorted(e for e in args.episodes_dir.iterdir() if e.is_dir() and episode_ok(e))
    if len(episodes) <= args.test_episodes:
        raise SystemExit(f"need more than {args.test_episodes} good episodes, found {len(episodes)}")
    test_eps = {e.name for e in episodes[-args.test_episodes:]}
    bin_dir = args.out / "velodyne_points" / "data"
    bin_dir.mkdir(parents=True, exist_ok=True)
    flip = np.array([1.0, 1.0, -1.0])

    frames, fills = [], []
    for ep in episodes:
        n0 = len(frames)
        if (ep / "l2_scans.npz").exists():
            sweeps = npz_sweeps(ep / "l2_scans.npz", args.messages_per_scan, args.stride, args.limit)
        else:
            sweeps = accumulate_sweeps(
                ep / "bag", args.topic, args.messages_per_scan, args.typestore, args.limit, args.stride
            )
        for xyz, _, _, stamp, span in sweeps:
            image = grid.project(xyz * flip, np.zeros(len(xyz)), args.min_depth, args.max_depth)
            mask = image[..., lidar_grid.CH_MASK]
            keep = mask > 0
            sid = len(frames)
            lidar_grid.save_scan(
                bin_dir / f"{sid:010d}.bin",
                image[keep][:, [lidar_grid.CH_X, lidar_grid.CH_Y, lidar_grid.CH_Z]],
                image[keep][:, lidar_grid.CH_REFLECTANCE],
            )
            fills.append(float(mask.mean()))
            frames.append({
                "sample_id": sid,
                "file_name": f"{sid:010d}.bin",
                "episode": ep.name,
                "t_ref": stamp,
                "window_seconds": span,
                "num_points": int(keep.sum()),
                "pixel_fill": fills[-1],
                "split": "test" if ep.name in test_eps else "train",
            })
        print(f"{ep.name}: {len(frames) - n0} sweeps", flush=True)

    lidar_grid.write_dataset_metadata(args.out, frames, {
        "source": "hospital sim, Unitree_L2 emulation (lidargen), /lidar_points",
        "episodes_dir": str(args.episodes_dir),
        "topic": args.topic,
        "num_rings": grid.height,
        "image_width": grid.width,
        "h_up_deg": grid.h_up,
        "h_down_deg": grid.h_down,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "intensity_scale": None,
        "ego_motion_compensated": False,
        "messages_per_scan": args.messages_per_scan,
        "stride": args.stride,
        "num_frames": len(frames),
        "num_train": sum(f["split"] == "train" for f in frames),
        "num_test": sum(f["split"] == "test" for f in frames),
        "test_episodes": sorted(test_eps),
    })
    fills = np.array(fills)
    spans = np.array([f["window_seconds"] for f in frames])
    print(f"wrote {len(frames)} sweeps ({len(test_eps)} test episodes) to {bin_dir}")
    print(f"  window: mean={spans.mean():.3f}s  pixel fill: mean={fills.mean():.4f} "
          f"min={fills.min():.4f} max={fills.max():.4f}")


if __name__ == "__main__":
    main()

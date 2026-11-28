#!/usr/bin/env python3
"""Extract base-link trajectories from the hospital-sim episodes for re-rendering.

For every episode under --episodes-dir, this reads /state_estimation
(nav_msgs/Odometry, map -> base, the same as world) and writes
<out>/<episode>/poses.npz with t (s), pos (N, 3) and quat_wxyz (N, 4). It also
copies episode.json. Feed the output directories to
../lidargen: `python -m lidargen.scripts.replay_poses_l2 --episode <dir>`.

    python tools/extract_sim_poses.py --episodes-dir r2flow/data/rosbags/simulated_bags \\
        --out r2flow/data/rosbags/simulated_l2_replay
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-dir", type=Path, default=Path("r2flow/data/rosbags/simulated_bags"))
    p.add_argument("--out", type=Path, default=Path("r2flow/data/rosbags/simulated_l2_replay"))
    p.add_argument("--topic", default="/state_estimation")
    args = p.parse_args(argv)
    store = get_typestore(Stores.ROS2_HUMBLE)
    for ep in sorted(d for d in args.episodes_dir.iterdir() if (d / "bag").is_dir()):
        with AnyReader([ep / "bag"], default_typestore=store) as reader:
            con = [c for c in reader.connections if c.topic == args.topic]
            t, pos, quat = [], [], []
            for c, _, raw in reader.messages(connections=con):
                m = reader.deserialize(raw, c.msgtype)
                t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                pp, q = m.pose.pose.position, m.pose.pose.orientation
                pos.append((pp.x, pp.y, pp.z))
                quat.append((q.w, q.x, q.y, q.z))
        t = np.asarray(t)
        order = np.argsort(t, kind="stable")
        t, keep = np.unique(t[order], return_index=True)
        pos = np.asarray(pos)[order][keep]
        quat = np.asarray(quat)[order][keep]
        dst = args.out / ep.name
        dst.mkdir(parents=True, exist_ok=True)
        np.savez(dst / "poses.npz", t=t, pos=pos, quat_wxyz=quat)
        if (ep / "episode.json").exists():
            shutil.copy(ep / "episode.json", dst / "episode.json")
        print(f"{ep.name}: {len(t)} poses over {t[-1] - t[0]:.1f}s, base z median {np.median(pos[:, 2]):.3f}")


if __name__ == "__main__":
    main()

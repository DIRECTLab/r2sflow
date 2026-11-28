#!/usr/bin/env python3
"""Run the conditional real->sim model on a live (or replayed) Go2 utlidar stream.

Pipeline per output scan, the same as training data preparation
(tools/rosbag2_go2_to_r2flow.py): accumulate `--messages-per-scan` utlidar
messages in a sliding window, flip z (the utlidar frame is z-down), project onto
the shared -24..0 deg grid, translate with the model, then unproject and flip z
back so the output cloud is in the input's frame.

Two front ends share `Real2SimRunner`:

  ROS 2 node (needs rclpy *and* torch in one environment):
      python tools/real2sim_node.py ros --ckpt <cond .pth>
      subscribes /go2/raw_lidar, publishes /real2sim/cloud and /real2sim/input

  Offline replay (no ROS install; rosbags + torch): reads a bag, runs the model
  on every sweep, writes a rosbag2 with the input and output clouds for RViz,
  and reports latency.
      python tools/real2sim_node.py bag --ckpt <cond .pth> \
          --bag r2flow/data/rosbags/go2_mocam_loop --out-bag logs/real2sim_replay
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import r2flow.utils.flow as flow  # noqa: E402
from lidar_grid import CH_DEPTH, RayGrid  # noqa: E402
from r2flow.utils.inference import setup_model  # noqa: E402

# Ray grid per training dataset; must match r2flow.utils.lidar.SPHERICAL_RAY_ANGLES.
GRIDS = {
    "go2w_pseudoreal": RayGrid(64, 1024, h_up=0.0, h_down=-24.0),
    "go2w_common_sim": RayGrid(64, 1024, h_up=0.0, h_down=-24.0),
    "go2w_l2_sim": RayGrid(64, 1024, h_up=0.0, h_down=-24.0),
}


class Real2SimRunner:
    def __init__(self, ckpt, steps=4, device="cuda", bf16=True,
                 messages_per_scan=8, stride=2, flip_z=True, seed=0):
        self.device = torch.device(device)
        self.model, self.lidar_utils, self.cfg = setup_model(ckpt, device=self.device)
        assert self.model.cond_channels == 1, "expected a conditional model"
        self.grid = GRIDS[self.cfg.data.dataset]
        assert (self.grid.height, self.grid.width) == tuple(self.cfg.data.resolution)
        self.steps, self.bf16, self.flip_z = steps, bf16, flip_z
        self.window = collections.deque(maxlen=messages_per_scan)
        self.stride = stride
        self.since_emit = 0
        self.emitted = 0
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.zsign = np.array([1.0, 1.0, -1.0 if flip_z else 1.0])

    def push(self, xyz: np.ndarray):
        """Add one message; returns (input_xyz, output_xyz, seconds) when a sweep is due."""
        self.window.append(xyz)
        self.since_emit += 1
        if len(self.window) < self.window.maxlen or (self.emitted and self.since_emit < self.stride):
            return None
        self.since_emit = 0
        self.emitted += 1
        return self.process(np.concatenate(self.window))

    def depth_image(self, xyz: np.ndarray) -> torch.Tensor:
        image = self.grid.project(
            xyz * self.zsign, np.zeros(len(xyz)), self.cfg.data.min_depth, self.cfg.data.max_depth
        )
        return torch.from_numpy(image[..., CH_DEPTH])[None, None].to(self.device)

    def to_points(self, depth: torch.Tensor) -> np.ndarray:
        xyz = self.lidar_utils.convert_metric_depth(depth, format="cartesian")[0]
        keep = (depth[0, 0] > 0).flatten()
        pts = xyz.flatten(1).T[keep].cpu().numpy().astype(np.float64)
        return pts * self.zsign  # back to the sensor's own frame

    @torch.no_grad()
    def translate(self, depth: torch.Tensor) -> torch.Tensor:
        lu, cfg = self.lidar_utils, self.cfg
        cond = flow.encode(lu, depth, None, cfg.data.data_format, False)
        noise = torch.randn(cond.shape, generator=self.generator, device=self.device)
        x = cond + cfg.flow.bridge_sigma * noise if cfg.flow.source == "bridge" else noise

        def velocity(t, x):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                return self.model(t, x, cond=cond).float()

        x = flow.euler_trajectory(velocity, x, num_steps=self.steps)[-1]
        return flow.decode(lu, x, cfg.data.data_format, False)[0]

    def process(self, xyz: np.ndarray):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        depth = self.depth_image(xyz)
        out = self.translate(depth)
        out_xyz = self.to_points(out)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return self.to_points(depth), out_xyz, time.perf_counter() - t


# ------------------------------------------------------------------ PointCloud2
def xyz_fields(PointField):
    return [PointField(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyz")]


def make_cloud(types, xyz: np.ndarray, frame_id: str, stamp_ns: int):
    """sensor_msgs/PointCloud2 (float32 x,y,z) for a rosbags typestore."""
    Header, Time = types["std_msgs/msg/Header"], types["builtin_interfaces/msg/Time"]
    PointCloud2, PointField = types["sensor_msgs/msg/PointCloud2"], types["sensor_msgs/msg/PointField"]
    data = np.ascontiguousarray(xyz, dtype=np.float32)
    return PointCloud2(
        header=Header(stamp=Time(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9), frame_id=frame_id),
        height=1, width=len(data), fields=xyz_fields(PointField), is_bigendian=False,
        point_step=12, row_step=12 * len(data), data=data.view(np.uint8).reshape(-1),
        is_dense=True,
    )


def run_bag(args):
    import bag_io
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    runner = Real2SimRunner(args.ckpt, args.steps, args.device, args.bf16,
                            args.messages_per_scan, args.stride, not args.no_flip_z)
    store = get_typestore(Stores.ROS2_HUMBLE)
    frame_id = args.frame_id
    latencies, fills = [], []
    out_bag = Path(args.out_bag)
    if out_bag.exists():
        raise SystemExit(f"{out_bag} exists; pick a new --out-bag")
    out_bag.parent.mkdir(parents=True, exist_ok=True)
    with Writer(out_bag, version=8) as writer:
        conn_in = writer.add_connection("/real2sim/input", "sensor_msgs/msg/PointCloud2", typestore=store)
        conn_out = writer.add_connection("/real2sim/cloud", "sensor_msgs/msg/PointCloud2", typestore=store)
        for i, (stamp, points) in enumerate(bag_io.iter_pointclouds([Path(args.bag)], args.topic)):
            if args.start_message is not None and i < args.start_message:
                continue
            xyz, _ = bag_io.as_xyzi(points)
            result = runner.push(xyz)
            if result is None:
                continue
            in_xyz, out_xyz, sec = result
            if runner.emitted > args.warmup:
                latencies.append(sec)
            fills.append((len(in_xyz), len(out_xyz)))
            ns = int(stamp * 1e9)
            for conn, pts in ((conn_in, in_xyz), (conn_out, out_xyz)):
                msg = make_cloud(store.types, pts, frame_id, ns)
                writer.write(conn, ns, store.serialize_cdr(msg, "sensor_msgs/msg/PointCloud2"))
            if args.limit and runner.emitted >= args.limit:
                break
    lat = np.array(latencies) if latencies else np.zeros(1)
    report = {
        "ckpt": str(args.ckpt), "steps": args.steps, "bf16": args.bf16,
        "sweeps": runner.emitted,
        "latency_ms": {"mean": 1e3 * lat.mean(), "p50": 1e3 * np.percentile(lat, 50),
                       "p95": 1e3 * np.percentile(lat, 95), "max": 1e3 * lat.max()},
        "points_in_mean": float(np.mean([f[0] for f in fills])),
        "points_out_mean": float(np.mean([f[1] for f in fills])),
        "sweep_period_ms": 1e3 * args.stride * 0.065,  # utlidar message ~65 ms
        "out_bag": str(out_bag),
    }
    (out_bag.parent / f"{out_bag.name}_latency.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    runner = Real2SimRunner(args.ckpt, args.steps, args.device, args.bf16,
                            args.messages_per_scan, args.stride, not args.no_flip_z)

    def to_msg(xyz, header):
        data = np.ascontiguousarray(xyz, dtype=np.float32)
        return PointCloud2(
            header=Header(stamp=header.stamp, frame_id=header.frame_id), height=1,
            width=len(data), fields=xyz_fields(PointField), is_bigendian=False,
            point_step=12, row_step=12 * len(data), data=data.tobytes(), is_dense=True,
        )

    class Real2SimNode(Node):
        def __init__(self):
            super().__init__("real2sim")
            self.pub_out = self.create_publisher(PointCloud2, "/real2sim/cloud", 2)
            self.pub_in = self.create_publisher(PointCloud2, "/real2sim/input", 2)
            self.create_subscription(PointCloud2, args.topic, self.callback, qos_profile_sensor_data)

        def callback(self, msg):
            sys.path.insert(0, str(ROOT / "tools"))
            from bag_io import as_xyzi, decode_pointcloud

            xyz, _ = as_xyzi(decode_pointcloud(msg))
            result = runner.push(xyz)
            if result is None:
                return
            in_xyz, out_xyz, sec = result
            self.pub_in.publish(to_msg(in_xyz, msg.header))
            self.pub_out.publish(to_msg(out_xyz, msg.header))
            self.get_logger().info(f"real2sim {len(in_xyz)} -> {len(out_xyz)} pts in {1e3 * sec:.1f} ms",
                                   throttle_duration_sec=2.0)

    rclpy.init()
    rclpy.spin(Real2SimNode())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("ros", "bag"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--topic", default="/go2/raw_lidar")
    p.add_argument("--messages-per-scan", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--no-flip-z", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bag", type=Path, help="bag mode: input rosbag2")
    p.add_argument("--out-bag", type=Path, help="bag mode: output rosbag2 directory")
    p.add_argument("--frame-id", default="go2/utlidar_lidar")
    p.add_argument("--start-message", type=int, default=None,
                   help="bag mode: skip to this message (e.g. the held-out tail)")
    p.add_argument("--limit", type=int, default=0, help="bag mode: stop after N sweeps")
    p.add_argument("--warmup", type=int, default=3, help="sweeps excluded from latency")
    args = p.parse_args(argv)
    run_bag(args) if args.mode == "bag" else run_ros(args)


if __name__ == "__main__":
    main()

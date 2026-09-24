"""Rosbag reading helpers shared by the converters.

Requires `rosbags`; keep torch-side code out of here. `AnyReader` handles both
ROS 1 `.bag` directories/files and ROS 2 `.db3` bags, but rosbag2 metadata v5
does not embed message definitions, so a typestore has to be supplied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# sensor_msgs/PointField datatype enum -> numpy dtype
PF_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def open_reader(paths: list[Path], typestore: str = "ros2_humble") -> AnyReader:
    """AnyReader with a fallback typestore for bags lacking type definitions."""
    store = {
        "ros2_humble": Stores.ROS2_HUMBLE,
        "ros2_jazzy": Stores.ROS2_JAZZY,
        "ros1_noetic": Stores.ROS1_NOETIC,
    }[typestore]
    return AnyReader(paths, default_typestore=get_typestore(store))


def stamp_seconds(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def decode_pointcloud(msg) -> np.ndarray:
    """PointCloud2 -> structured array over its own fields (zero-copy view)."""
    if msg.is_bigendian:
        raise ValueError("big-endian PointCloud2 is not supported")
    fields = [f for f in msg.fields if f.datatype in PF_DTYPE]
    dtype = np.dtype(
        {
            "names": [f.name for f in fields],
            "formats": [PF_DTYPE[f.datatype] for f in fields],
            "offsets": [f.offset for f in fields],
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)


def as_xyzi(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Structured array -> (xyz, intensity) float64, finite rows only."""
    names = points.dtype.names
    intensity = (
        points["intensity"] if "intensity" in names else np.zeros(len(points))
    )
    xyz = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64)
    intensity = intensity.astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
    return xyz[finite], intensity[finite]


def field_or_none(points: np.ndarray, name: str) -> np.ndarray | None:
    return points[name].astype(np.float64) if name in points.dtype.names else None


def list_topics(paths: list[Path], typestore: str = "ros2_humble") -> list[tuple[str, str, int]]:
    with open_reader(paths, typestore) as reader:
        return [(c.topic, c.msgtype, c.msgcount) for c in reader.connections]


def iter_pointclouds(paths: list[Path], topic: str, typestore: str = "ros2_humble"):
    """Yield (stamp_seconds, structured_points) for every PointCloud2 on `topic`."""
    with open_reader(paths, typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic!r} not in bag. available: {available}")
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            yield stamp_seconds(msg.header), decode_pointcloud(msg)

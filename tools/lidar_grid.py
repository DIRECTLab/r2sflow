"""Ray-grid geometry shared by the rosbag converters and the visualiser.

Pure numpy on purpose: the bag tools run in a light `rosbags`+`numpy` venv while
the visualiser runs inside the torch container, and both need this projection.
Importing torch here would break the former.

The grid convention matches `r2flow.utils.lidar.get_linear_ray_angles`, so a
range image built here lines up cell-for-cell with `model.coords`:

    elevation(row) = (1 - row / H) * (h_up - h_down) + h_down   # row 0 = topmost
    azimuth(col)   = 180 - 360 * col / W                        # col 0 = +180 deg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channel layout of a projected scan, matching r2flow/data/*/*.py
CH_X, CH_Y, CH_Z, CH_REFLECTANCE, CH_DEPTH, CH_MASK = range(6)
NUM_CHANNELS = 6


@dataclass(frozen=True)
class RayGrid:
    """A linear elevation/azimuth ray grid."""

    height: int
    width: int
    h_up: float
    h_down: float

    @property
    def elevation_step(self) -> float:
        return (self.h_up - self.h_down) / self.height

    @property
    def azimuth_step(self) -> float:
        return 360.0 / self.width

    def elevations(self) -> np.ndarray:
        """Per-row ray elevation in degrees, row 0 topmost."""
        return (1 - np.arange(self.height) / self.height) * (
            self.h_up - self.h_down
        ) + self.h_down

    def row_col(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest ray cell for each point. Returns (row, col, inside_band)."""
        depth = np.linalg.norm(xyz, axis=1)
        safe = np.maximum(depth, 1e-9)
        elevation = np.degrees(np.arcsin((xyz[:, 2] / safe).clip(-1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        row = np.rint((self.h_up - elevation) / self.elevation_step).astype(np.int64)
        col = np.mod(np.rint((180.0 - azimuth) / self.azimuth_step).astype(np.int64), self.width)
        inside = (row >= 0) & (row < self.height) & (depth > 0)
        return row, col, inside

    def project(
        self,
        xyz: np.ndarray,
        reflectance: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> np.ndarray:
        """Project a point cloud to an (H, W, 6) image; the closest point wins.

        Points whose elevation falls outside the grid's band are dropped rather
        than clamped onto the first/last row.
        """
        depth = np.linalg.norm(xyz, axis=1)
        row, col, inside = self.row_col(xyz)
        keep = inside & (depth >= min_depth) & (depth <= max_depth)
        xyz, reflectance, depth = xyz[keep], reflectance[keep], depth[keep]
        row, col = row[keep], col[keep]

        image = np.zeros((self.height, self.width, NUM_CHANNELS), dtype=np.float32)
        # Write far points first so nearer ones overwrite them.
        order = np.argsort(-depth)
        image[row[order], col[order], CH_X] = xyz[order, 0]
        image[row[order], col[order], CH_Y] = xyz[order, 1]
        image[row[order], col[order], CH_Z] = xyz[order, 2]
        image[row[order], col[order], CH_REFLECTANCE] = reflectance[order]
        image[row[order], col[order], CH_DEPTH] = depth[order]
        image[row[order], col[order], CH_MASK] = 1.0
        return image


# The grid the go2w_sim checkpoints were trained on. Mirrors
# r2flow.utils.lidar.get_go2w_livox_linear_ray_angles(40, 512).
MODEL_GRID = RayGrid(height=40, width=512, h_up=29.50, h_down=-29.70)


def diagnose_origin(
    xyz: np.ndarray,
    sweep_time: np.ndarray | None = None,
    candidates: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Sanity-check that a cloud really is centred on its sensor.

    A spherical scan seen from its true origin has (a) an empty near-field void
    of about the sensor's minimum range, and (b) if per-point timestamps are
    available, a smoothly advancing elevation as the sensor sweeps. Viewed from
    a wrong origin the void collapses and the elevation trace tears.

    Returns one entry per candidate; the best origin has the largest
    `void_radius` together with the smallest `elevation_step_p99`.
    """
    if candidates is None:
        candidates = {
            "(0,0,0)": np.zeros(3),
            "(0,0,+0.345)": np.array([0.0, 0.0, 0.345]),
            "(0,0,-0.345)": np.array([0.0, 0.0, -0.345]),
        }
    order = np.argsort(sweep_time) if sweep_time is not None else None

    report = {}
    for name, origin in candidates.items():
        p = xyz - origin
        depth = np.linalg.norm(p, axis=1)
        entry = {"void_radius": float(depth.min())}
        if order is not None:
            q = p[order]
            d = np.maximum(np.linalg.norm(q, axis=1), 1e-9)
            elevation = np.degrees(np.arcsin((q[:, 2] / d).clip(-1, 1)))
            steps = np.abs(np.diff(elevation))
            entry["elevation_step_p99"] = float(np.percentile(steps, 99))
        report[name] = entry
    return report


def save_scan(path: Path, xyz: np.ndarray, reflectance: np.ndarray) -> None:
    """Write a KITTI-style float32 [x, y, z, reflectance] scan."""
    scan = np.empty((len(xyz), 4), dtype=np.float32)
    scan[:, :3] = xyz
    scan[:, 3] = reflectance
    scan.tofile(path)


def load_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a KITTI-style scan back as (xyz, reflectance)."""
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3].astype(np.float64), scan[:, 3].astype(np.float64)


def write_dataset_metadata(out_dir: Path, frames: list[dict], sensor_info: dict) -> None:
    (out_dir / "frames.json").write_text(json.dumps(frames, indent=1))
    (out_dir / "sensor_info.json").write_text(json.dumps(sensor_info, indent=1))


def grid_from_sensor_info(info: dict) -> RayGrid:
    """Rebuild the grid a dataset was written against."""
    return RayGrid(
        height=info["num_rings"],
        width=info.get("image_width", MODEL_GRID.width),
        h_up=info["h_up_deg"],
        h_down=info["h_down_deg"],
    )

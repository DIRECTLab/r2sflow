import json
from pathlib import Path

import datasets as ds
import numba
import numpy as np

_DESCRIPTION = """
LiDAR scans from simulated navigation episodes, converted from `/registered_scan`
in ROS 2 bags by `tools/ros2bag_registered_to_r2flow.py`. Each message is a full
scan registered into `map`; the converter moves it back into the sensor frame
with `map -> sensor_at_scan`. The topic has no intensity, so reflectance is zero.

The scans span about -24..+11 deg of elevation. The grid is 64 x 1024, the
KITTI-360 resolution, so the pretrained R2Flow HDiT loads without resizing its
positional embedding.
"""

# Ray grid, read back from the dataset and asserted below so a converter change
# cannot silently desync from r2flow.utils.lidar.get_go2w_nav_linear_ray_angles.
H_UP = 12.0
H_DOWN = -24.0
NUM_RINGS = 64

_CONFIGS = [("spherical-64x1024", 64, 1024, "-24..+12 deg, 0.5625 deg rows")]


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(
    point_path: str,
    H: int = 64,
    W: int = 1024,
    min_depth: float = 0.5,
    max_depth: float = 30.0,
):
    """Project a sensor-frame scan onto the ray grid (row 0 = H_UP)."""
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    # Nearest ray cell. Row 0 is H_UP and rows descend to H_DOWN;
    # column 0 is azimuth +180 deg advancing clockwise, matching
    # get_linear_ray_angles.
    elevation = np.degrees(np.arcsin(z / np.maximum(depth, 1e-9)))
    grid_h = np.rint((H_UP - elevation) / ((H_UP - H_DOWN) / H)).astype(np.int32)

    azimuth = np.degrees(np.arctan2(y, x))
    grid_w = np.rint((180.0 - azimuth) / (360.0 / W)).astype(np.int32) % W

    inside = ((grid_h >= 0) & (grid_h < H)).squeeze(1)
    grid = np.concatenate((grid_h, grid_w), axis=1)[inside]
    points = points[inside]
    depth = depth[inside]

    # projection, closest point wins
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)


class Go2WNavSim(ds.GeneratorBasedBuilder):
    """Simulated navigation scans from ROS 2 `/registered_scan` bags"""

    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name=name,
            description=f"spherical projection, {h}x{w} ({note})",
            data_dir="r2flow/data/go2w_nav_sim/dataset",
        )
        for name, h, w, note in _CONFIGS
    ]

    DEFAULT_CONFIG_NAME = "spherical-64x1024"

    def _parse_config_name(self):
        _, resolution = self.config.name.split("-")
        height, width = resolution.split("x")
        return int(height), int(width)

    def _sensor_info(self):
        info_path = Path(self.config.data_dir) / "sensor_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} not found. Build the dataset first with "
                "`python tools/ros2bag_registered_to_r2flow.py --episodes-dir <episodes dir>`."
            )
        info = json.loads(info_path.read_text())
        if (info["h_up_deg"], info["h_down_deg"], info["num_rings"]) != (
            H_UP, H_DOWN, NUM_RINGS
        ):
            raise ValueError(
                f"{info_path} describes a different ray grid "
                f"(h_up={info['h_up_deg']}, h_down={info['h_down_deg']}, "
                f"rings={info['num_rings']}) than this builder assumes "
                f"(h_up={H_UP}, h_down={H_DOWN}, rings={NUM_RINGS})."
            )
        return info

    def _info(self):
        height, width = self._parse_config_name()
        features = {
            "sample_id": ds.Value("int32"),
            "xyz": ds.Array3D((3, height, width), "float32"),
            "reflectance": ds.Array3D((1, height, width), "float32"),
            "depth": ds.Array3D((1, height, width), "float32"),
            "mask": ds.Array3D((1, height, width), "float32"),
        }
        return ds.DatasetInfo(description=_DESCRIPTION, features=ds.Features(features))

    def _split_generators(self, _):
        info = self._sensor_info()
        data_dir = Path(self.config.data_dir)
        frames = json.loads((data_dir / "frames.json").read_text())
        splits = list()
        for split, tag in ((ds.Split.TRAIN, "train"), (ds.Split.TEST, "test")):
            items = [
                (f["sample_id"], data_dir / "velodyne_points" / "data" / f["file_name"])
                for f in frames
                if f["split"] == tag
            ]
            splits.append(
                ds.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "items": items,
                        "min_depth": info["min_depth"],
                        "max_depth": info["max_depth"],
                    },
                )
            )
        return splits

    def _generate_examples(self, items, min_depth, max_depth):
        height, width = self._parse_config_name()
        for sample_id, file_path in items:
            xyzrdm = load_points_as_images(
                file_path, H=height, W=width, min_depth=min_depth, max_depth=max_depth
            )
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            xyzrdm *= xyzrdm[[5]]
            yield sample_id, {
                "sample_id": sample_id,
                "xyz": xyzrdm[:3],
                "reflectance": xyzrdm[[3]],
                "depth": xyzrdm[[4]],
                "mask": xyzrdm[[5]],
            }

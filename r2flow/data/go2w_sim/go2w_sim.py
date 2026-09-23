import json
from pathlib import Path

import datasets as ds
import numba
import numpy as np

_DESCRIPTION = """
LiDAR scans from a Unitree Go2-W simulator rosbag, assembled by
`tools/rosbag_to_r2flow.py`. The simulated Livox publishes one ~60 deg azimuth
sector per message, so each scan here is several consecutive messages de-skewed
into the sensor pose at the window's reference time and merged into a full turn.
"""

# Sensor ray grid, measured from the bags and exact to ~1e-5 deg:
#   elevation(ring) = 29.50 - 1.48 * ring,  ring in [0, 39]
# which is the repo's linear form with these bounds, so this grid agrees with
# r2flow.utils.lidar.get_go2w_livox_linear_ray_angles cell-for-cell.
# Kept in sync with tools/rosbag_to_r2flow.py, which asserts the same values.
H_UP = 29.50
H_DOWN = -29.70
NUM_RINGS = 40

# 40x512 is the default rather than the sensor's native 40x500 because every
# architecture here downsamples by 8: HDiT tokenizes by patch_size=(1, 4) and then
# halves three times (so it needs H % 8 == 0 and W % 32 == 0), and both U-Nets
# fail a skip-connection size check at width 500. 512 columns are 0.703 deg wide
# against the native 0.72 deg, so a few columns stay empty and the mask records it.
_CONFIGS = [
    ("spherical-40x512", 40, 512, "trainable: 512 is divisible by the 8x downsampling"),
    ("spherical-40x500", 40, 500, "native 0.72 deg grid, for analysis -- will not train"),
]


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(
    point_path: str,
    H: int = 40,
    W: int = 512,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
):
    """Project a de-skewed scan onto the sensor's ray grid.

    Unlike the KITTI-360 loader there is no scan-unfolding variant: the sim
    reports an exact ring index per point, so the elevation table is known and a
    nearest-cell spherical projection is already lossless up to de-skewing.
    """
    # load xyz & reflectance and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    # Nearest ray cell, so the sensor's rays land on cell centres. Row 0 is the
    # topmost ring and column 0 is azimuth +180 deg advancing clockwise, which is
    # the convention get_hdl64e_linear_ray_angles uses.
    elevation = np.degrees(np.arcsin(z / np.maximum(depth, 1e-9)))
    grid_h = np.rint((H_UP - elevation) / ((H_UP - H_DOWN) / H)).astype(np.int32)

    azimuth = np.degrees(np.arctan2(y, x))
    grid_w = np.rint((180.0 - azimuth) / (360.0 / W)).astype(np.int32) % W

    # De-skewing moves points off their original ring, so a few can land outside
    # the sensor's elevation band. Drop them rather than piling them on row 0/39.
    inside = ((grid_h >= 0) & (grid_h < H)).squeeze(1)
    grid = np.concatenate((grid_h, grid_w), axis=1)[inside]
    points = points[inside]
    depth = depth[inside]

    # projection, closest point wins
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)


class Go2WSim(ds.GeneratorBasedBuilder):
    """Unitree Go2-W simulator LiDAR scans"""

    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name=name,
            description=f"spherical projection, {h}x{w} resolution ({note})",
            data_dir="r2flow/data/go2w_sim/dataset",
        )
        for name, h, w, note in _CONFIGS
    ]

    DEFAULT_CONFIG_NAME = "spherical-40x512"

    def _parse_config_name(self):
        _, resolution = self.config.name.split("-")
        height, width = resolution.split("x")
        return int(height), int(width)

    def _sensor_info(self):
        info_path = Path(self.config.data_dir) / "sensor_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} not found. Build the dataset first with "
                "`python tools/rosbag_to_r2flow.py --bag-dir <rosbag dir>`."
            )
        info = json.loads(info_path.read_text())
        # Catch a ray grid that no longer matches what this builder projects onto.
        if (info["h_up_deg"], info["h_down_deg"], info["num_rings"]) != (H_UP, H_DOWN, NUM_RINGS):
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
        return ds.DatasetInfo(
            description=_DESCRIPTION, features=ds.Features(features)
        )

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

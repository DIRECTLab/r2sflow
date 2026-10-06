from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_linear_ray_angles(
    H: int,
    W: int,
    h_up: float,
    h_down: float,
    w_left: float = 180,
    w_right: float = -180,
    device: torch.device = "cpu",
):
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [h_down, h_up]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles


def get_hdl64e_linear_ray_angles(
    H: int = 64, W: int = 2048, device: torch.device = "cpu"
):
    return get_linear_ray_angles(H, W, h_up=3, h_down=-25, device=device)


def get_go2w_livox_linear_ray_angles(
    H: int = 40, W: int = 512, device: torch.device = "cpu"
):
    """Ray angles of the simulated Livox on the Unitree Go2-W.

    Measured from the rosbags: 40 rings at `elevation = 29.50 - 1.48 * ring`, so
    the linear form above reproduces the ring table exactly when H is 40. Kept in
    sync with `tools/rosbag_to_r2flow.py` and `r2flow/data/go2w_sim/go2w_sim.py`.
    """
    return get_linear_ray_angles(H, W, h_up=29.50, h_down=-29.70, device=device)


def get_go2w_l1_linear_ray_angles(
    H: int = 64, W: int = 512, device: torch.device = "cpu"
):
    """Ray angles of the simulated Unitree L1 on the Go2-W.

    The L1's `ring` field is a scan-order index, not an elevation row, so unlike
    the Livox there is no measured table to match -- this is the chosen grid.
    Only the lower hemisphere is covered: the front-mounted sensor faces down and
    the real utlidar reports nothing above the horizon. Kept in sync with
    `tools/rosbag_to_r2flow.py` (--sensor l1) and `r2flow/data/go2w_sim_l1`.
    """
    return get_linear_ray_angles(H, W, h_up=0.0, h_down=-90.0, device=device)


def get_go2w_nav_linear_ray_angles(
    H: int = 64, W: int = 1024, device: torch.device = "cpu"
):
    """Ray angles of the simulated navigation scans (`/registered_scan`).

    Measured in the sensor frame, the scans span about -24..+11 deg. 64 x 1024
    matches KITTI-360 so the pretrained R2Flow weights load as-is. Kept in sync
    with `tools/ros2bag_registered_to_r2flow.py` and `r2flow/data/go2w_nav_sim`.
    """
    return get_linear_ray_angles(H, W, h_up=12.0, h_down=-24.0, device=device)


# Spherical projection geometry per dataset, keyed by `cfg.data.dataset`.
SPHERICAL_RAY_ANGLES = {
    "kitti_raw": get_hdl64e_linear_ray_angles,
    "kitti_360": get_hdl64e_linear_ray_angles,
    "go2w_sim": get_go2w_livox_linear_ray_angles,
    "go2w_sim_l1": get_go2w_l1_linear_ray_angles,
    "go2w_nav_sim": get_go2w_nav_linear_ray_angles,
}


class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution: tuple[int, int],
        format: Literal["logscale", "inverse", "metric", "cartesian"],
        min_depth: float,
        max_depth: float,
        ray_angles: torch.Tensor = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.format = format
        self.min_depth = min_depth
        self.max_depth = max_depth
        if ray_angles is None:
            ray_angles = get_hdl64e_linear_ray_angles(*resolution)
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=self.resolution,
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles.float())

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [-1, +1] to [0, 1]"""
        return ((x + 1) / 2).clamp(0, 1)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [0, 1] to [-1, +1]"""
        return (x * 2 - 1).clamp(-1, 1)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()

    @torch.no_grad()
    def convert_metric_depth(
        self,
        metric_depth: torch.Tensor,
        mask: torch.Tensor | None = None,
        format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [-1, 1].
        """
        if format is None:
            format = self.format
        if mask is None:
            mask = self.get_mask(metric_depth)
        if format == "logscale":
            converted_depth = torch.log2(metric_depth + 1) / np.log2(self.max_depth + 1)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "inverse":
            converted_depth = self.min_depth / metric_depth.add(1e-8)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "metric":
            converted_depth = metric_depth.div(self.max_depth)
            converted_depth = self.normalize(converted_depth * mask)
        elif format == "cartesian":
            """metric -> xyz is irreversible if spherical projection"""
            phi = self.ray_angles[:, [0]]
            theta = self.ray_angles[:, [1]]
            grid_x = metric_depth * phi.cos() * theta.cos()
            grid_y = metric_depth * phi.cos() * theta.sin()
            grid_z = metric_depth * phi.sin()
            converted_depth = torch.cat((grid_x, grid_y, grid_z), dim=1) * mask
        else:
            raise ValueError("Invalid depth format")
        return converted_depth

    @torch.no_grad()
    def restore_metric_depth(
        self,
        converted_depth: torch.Tensor,
        format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [-1, 1] back to metric depth in [0, `max_depth`].
        """
        if format is None:
            format = self.format
        if format == "logscale":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = torch.exp2(converted_depth * np.log2(self.max_depth + 1)) - 1
        elif format == "inverse":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = self.min_depth / converted_depth.add(1e-8)
        elif format == "metric":
            converted_depth = self.denormalize(converted_depth)
            metric_depth = converted_depth.mul(self.max_depth)
        elif format == "cartesian":
            converted_depth = converted_depth * self.max_depth
            metric_depth = torch.norm(converted_depth, dim=1, p=2, keepdim=True)
        else:
            raise ValueError
        return metric_depth * self.get_mask(metric_depth)

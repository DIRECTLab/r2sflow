#!/usr/bin/env python3
"""Visualise how the flow transports a real LiDAR scan, step by step.

Takes scans produced by either rosbag converter, feeds each one to the model as
the ODE's starting state, and Euler-integrates it forward. Every intermediate
state is kept, so one figure shows t=0 (the untouched input) through t=N (the
final output) as both range images and bird's-eye point clouds.

Because R2Flow is an unconditional generative model, this is transport rather
than reconstruction: the further along the trajectory, the more the scan is
reshaped toward the distribution the checkpoint was trained on.

Run inside the container:

    python tools/visualize_flow.py \
        --ckpt logs/.../models/checkpoint_0002560000.pth \
        --scans /mnt/fast/lidar_data/go2_real_r2flow \
        --num-samples 4 --out logs/flow_viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import einops
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lidar_grid
import r2flow.utils


# =====================================================================================
# Loading scans
# =====================================================================================
def load_scan_batch(
    scans_dir: Path, indices: list[int], grid: lidar_grid.RayGrid, info: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Project the chosen .bin scans onto `grid` -> (depth, reflectance, mask)."""
    frames = json.loads((scans_dir / "frames.json").read_text())
    data_dir = scans_dir / "velodyne_points" / "data"

    depths, reflectances, masks, names = [], [], [], []
    for i in indices:
        frame = frames[i]
        xyz, reflectance = lidar_grid.load_scan(data_dir / frame["file_name"])
        image = grid.project(xyz, reflectance, info["min_depth"], info["max_depth"])
        depths.append(image[..., lidar_grid.CH_DEPTH])
        reflectances.append(image[..., lidar_grid.CH_REFLECTANCE])
        masks.append(image[..., lidar_grid.CH_MASK])
        names.append(frame["file_name"])

    to_t = lambda a: torch.from_numpy(np.stack(a)).float().unsqueeze(1)
    return to_t(depths), to_t(reflectances), to_t(masks), names


# =====================================================================================
# Rendering one state
# =====================================================================================
@torch.no_grad()
def render_state(
    state: torch.Tensor,
    lidar_utils,
    cfg,
    bev_size: int,
    depth_clip: float,
    bev_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A model tensor -> (range image RGB, BEV RGB), both HxWx3 uint8.

    Follows the same path as `log_images` in train.py, except that the display
    scales are `depth_clip` / `bev_range` metres rather than `max_depth`. The
    model's own 80 m normalisation is right for training but useless for looking
    at an indoor scan: an 8 m room would render entirely in the bottom 10% of the
    colour map and collapse to a few pixels in the BEV. This affects display
    only -- the tensor fed to the model is untouched.
    """
    range_image, _reflectance = r2flow.utils.flow.split_channels(
        state, cfg.data.data_format, cfg.data.train_reflectance
    )
    metric_depth = lidar_utils.restore_metric_depth(range_image)

    depth_rgb = r2flow.utils.render.colorize((metric_depth / depth_clip).clamp(0, 1))
    depth_rgb = einops.rearrange(depth_rgb[0], "c h w -> h w c").cpu().numpy()

    if cfg.data.data_format == "cartesian":
        xyz = range_image * lidar_utils.max_depth
    else:
        xyz = lidar_utils.convert_metric_depth(metric_depth, format="cartesian")
    normal = -r2flow.utils.render.estimate_surface_normal(xyz)
    normal = lidar_utils.denormalize(normal)
    bev = r2flow.utils.render.render_point_clouds(
        points=einops.rearrange(xyz / bev_range, "b c h w -> b (h w) c"),
        colors=einops.rearrange(normal, "b c h w -> b (h w) c"),
        t=torch.tensor([0, 0, 1.0]).to(xyz),
        size=bev_size,
    )
    bev_rgb = einops.rearrange(bev[0], "c h w -> h w c").clamp(0, 1).cpu().numpy()
    return depth_rgb, (bev_rgb * 255).astype(np.uint8)


# =====================================================================================
# Figure
# =====================================================================================
def make_figure(panels: list[tuple[np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    """Stack the range images and lay the BEVs out in a grid beneath them."""
    n = len(panels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Range images are ~12.8:1, so stack them; BEVs are square, so grid them.
    height, width = panels[0][0].shape[:2]
    fig_width = 16.0
    strip_height = n * fig_width / (width / height)
    grid_height = fig_width * rows / cols

    fig = plt.figure(figsize=(fig_width, strip_height + grid_height))
    # Explicit margins: the default figure padding would eat ~12% of the height
    # as blank space above the first range image.
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[strip_height, grid_height],
        hspace=0.10, top=0.965, bottom=0.015, left=0.07, right=0.99,
    )

    top = outer[0].subgridspec(n, 1, hspace=0.45)
    for i, (depth_rgb, _) in enumerate(panels):
        ax = fig.add_subplot(top[i])
        ax.imshow(depth_rgb, interpolation="nearest", aspect="auto")
        ax.set_ylabel(_label(i, n), rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    bottom = outer[1].subgridspec(rows, cols, hspace=0.18, wspace=0.05)
    for i, (_, bev_rgb) in enumerate(panels):
        ax = fig.add_subplot(bottom[i // cols, i % cols])
        ax.imshow(bev_rgb, interpolation="nearest")
        ax.set_title(_label(i, n), fontsize=9)
        ax.axis("off")
    for j in range(n, rows * cols):
        fig.add_subplot(bottom[j // cols, j % cols]).axis("off")

    fig.suptitle(title, fontsize=12, y=0.995)
    return fig


def _label(i: int, n: int) -> str:
    if i == 0:
        return "t=0\n(input)"
    if i == n - 1:
        return f"t={i}\n(output)"
    return f"t={i}"


# =====================================================================================
# Main
# =====================================================================================
def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--scans", type=Path, required=True,
                   help="dataset dir from either rosbag converter")
    p.add_argument("--out", type=Path, default=Path("logs/flow_viz"))
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-ids", type=int, nargs="+", default=None,
                   help="explicit frame indices (overrides --num-samples)")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--bev-size", type=int, default=320)
    p.add_argument("--depth-clip", type=float, default=None,
                   help="metres mapped to the top of the colour map "
                        "(default: 99th pct of the input depth)")
    p.add_argument("--bev-range", type=float, default=None,
                   help="half-extent of the BEV view in metres (default: --depth-clip)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.set_grad_enabled(False)
    model, lidar_utils, cfg = r2flow.utils.inference.setup_model(
        args.ckpt, device=args.device
    )

    info = json.loads((args.scans / "sensor_info.json").read_text())
    grid = lidar_grid.grid_from_sensor_info(info)
    if (grid.height, grid.width) != tuple(cfg.data.resolution):
        raise SystemExit(
            f"scan grid {grid.height}x{grid.width} != model resolution "
            f"{tuple(cfg.data.resolution)}; re-run the converter with "
            f"--height {cfg.data.resolution[0]} --width {cfg.data.resolution[1]}"
        )

    num_frames = len(json.loads((args.scans / "frames.json").read_text()))
    ids = args.sample_ids or np.linspace(
        0, num_frames - 1, min(args.num_samples, num_frames)
    ).astype(int).tolist()
    print(f"scans:   {args.scans}  ({num_frames} frames, using {ids})")

    depth, reflectance, mask, names = load_scan_batch(args.scans, ids, grid, info)
    print(f"input fill: {mask.mean():.4f} of {grid.height}x{grid.width}")

    # Scale the display to the input, not to the model's 80 m training range.
    observed = depth[mask > 0]
    depth_clip = args.depth_clip or float(
        np.ceil(torch.quantile(observed, 0.99).item())
    )
    bev_range = args.bev_range or depth_clip
    print(f"display: depth_clip={depth_clip:.1f} m  bev_range={bev_range:.1f} m "
          f"(input depth p50={observed.median():.2f} max={observed.max():.2f})")

    args.out.mkdir(parents=True, exist_ok=True)
    for k, (sample_id, name) in enumerate(zip(ids, names)):
        x = r2flow.utils.flow.encode(
            lidar_utils,
            depth[k : k + 1].to(args.device),
            reflectance[k : k + 1].to(args.device),
            cfg.data.data_format,
            cfg.data.train_reflectance,
        )
        states = r2flow.utils.flow.euler_trajectory(model, x, num_steps=args.num_steps)
        panels = [
            render_state(s, lidar_utils, cfg, args.bev_size, depth_clip, bev_range)
            for s in states
        ]

        fig = make_figure(
            panels,
            f"{args.scans.name}  frame {sample_id} ({name})   "
            f"{args.num_steps}-step Euler flow   ckpt={args.ckpt.name}   "
            f"depth 0-{depth_clip:.0f} m, BEV +-{bev_range:.0f} m",
        )
        path = args.out / f"flow_{sample_id:06d}.png"
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())

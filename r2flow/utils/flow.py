"""Integrating the learned rectified flow and keeping the intermediate states.

`sample.py` only needs the endpoint, so it hands the whole ODE to torchdiffeq.
Visualising the transport needs every step, hence this explicit Euler loop.
"""

from typing import Callable

import torch


@torch.no_grad()
def euler_trajectory(
    model: Callable,
    x: torch.Tensor,
    num_steps: int = 8,
    t_start: float = 0.0,
    t_end: float = 1.0,
    progress: Callable[[int], None] | None = None,
) -> list[torch.Tensor]:
    """Euler-integrate `dx/dt = model(t, x)` and return every state.

    Returns `num_steps + 1` tensors: the input at `t_start`, then one per step,
    ending at `t_end`. Feeding real data in as `x` rather than Gaussian noise
    transports it toward the model's learned distribution, which is what makes
    the intermediate states worth looking at.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    states = [x]
    dt = (t_end - t_start) / num_steps
    for step in range(num_steps):
        t = torch.full(
            (x.shape[0],), t_start + step * dt, device=x.device, dtype=x.dtype
        )
        x = x + dt * model(t, x)
        states.append(x)
        if progress is not None:
            progress(step + 1)
    return states


def encode(
    lidar_utils,
    depth: torch.Tensor,
    reflectance: torch.Tensor | None,
    data_format: str,
    train_reflectance: bool,
) -> torch.Tensor:
    """Metric depth (+ reflectance) -> the normalised tensor the model expects.

    Mirrors the `preprocess` step in train.py so that what we feed the model at
    visualisation time is byte-identical to what it saw during training.
    """
    channels = []
    if data_format == "cartesian":
        mask = lidar_utils.get_mask(depth)
        xyz = lidar_utils.convert_metric_depth(depth, format="cartesian")
        channels.append(xyz / lidar_utils.max_depth * mask)
    else:
        channels.append(lidar_utils.convert_metric_depth(depth))
    if train_reflectance:
        if reflectance is None:
            raise ValueError("model was trained with reflectance but none was given")
        channels.append(lidar_utils.normalize(reflectance))
    return torch.cat(channels, dim=1)


def split_channels(
    image: torch.Tensor, data_format: str, train_reflectance: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a model tensor back into (range, reflectance) parts."""
    sizes = [
        3 if data_format == "cartesian" else 1,
        1 if train_reflectance else 0,
    ]
    return torch.split(image, sizes, dim=1)

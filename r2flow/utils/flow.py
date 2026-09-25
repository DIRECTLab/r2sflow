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


def channel_sizes(data_format: str, train_reflectance: bool) -> list[int]:
    """[range channels, reflectance channels], as inference.py:26 computes them."""
    return [
        3 if data_format == "cartesian" else 1,
        1 if train_reflectance else 0,
    ]


def num_channels(data_format: str, train_reflectance: bool) -> int:
    return sum(channel_sizes(data_format, train_reflectance))


def split_channels(
    image: torch.Tensor, data_format: str, train_reflectance: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a model tensor back into (range, reflectance) parts."""
    return torch.split(image, channel_sizes(data_format, train_reflectance), dim=1)


@torch.no_grad()
def decode(
    lidar_utils,
    image: torch.Tensor,
    data_format: str,
    train_reflectance: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of `encode`: model tensor -> (metric depth, xyz, reflectance).

    Mirrors `postprocess` in sample.py, including its leading `clamp(-1, 1)`.
    Evaluation depends on that clamp: a generator's output is unconstrained and
    will contain values no real scan ever has, which would otherwise be a free
    tell for any feature extractor.
    """
    image = image.clamp(-1, 1)
    range_image, reflectance = split_channels(image, data_format, train_reflectance)
    metric_depth = lidar_utils.restore_metric_depth(range_image)
    reflectance = lidar_utils.denormalize(reflectance)
    if data_format == "cartesian":
        xyz = range_image * lidar_utils.max_depth
        xyz = xyz * lidar_utils.get_mask(metric_depth)
    else:
        xyz = lidar_utils.convert_metric_depth(metric_depth, format="cartesian")
    return metric_depth, xyz, reflectance


def make_flow_matcher(cfg):
    """The same flow matcher train.py builds, from a checkpoint's cfg.

    Deliberately not a hand-rolled `(1-t) * x_0 + t * x_1`: going through torchcfm
    means the forward process used at evaluation is the one the model was trained
    against by construction, including the `sigma` and formulation choices, rather
    than a copy that can silently drift.
    """
    import torchcfm.conditional_flow_matching as cfm

    if cfg.flow.formulation == "otcfm":
        return cfm.ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.flow.sigma)
    return cfm.ConditionalFlowMatcher(sigma=cfg.flow.sigma)


def noise_to_t0(
    flow_matcher, x_1: torch.Tensor, x_0: torch.Tensor, t0: float
) -> torch.Tensor:
    """Apply the training forward process at one fixed timestep.

    `t0=0` returns the noise untouched (the seed is fully discarded, so arms
    seeded from different data are then the same distribution); `t0=1` returns the
    data untouched.
    """
    t = torch.full((x_1.shape[0],), float(t0), device=x_1.device, dtype=x_1.dtype)
    _, x_t, _ = flow_matcher.sample_location_and_conditional_flow(x0=x_0, x1=x_1, t=t)
    return x_t

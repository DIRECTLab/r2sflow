"""Raydrop-aware guided sampling with a rectified flow prior (DRUM, adapted).

DRUM (Miyawaki et al., ICRA 2026, arXiv 2603.26263) makes "pseudo-real" scans from
clean simulated ones: a generative prior trained only on real scans is steered
toward a sim scan `y`, except where the prior wants to drop a ray, so the result
keeps the sim geometry (and therefore its labels) but gains real raydrop.

Here the prior is an R2Flow rectified flow (t=0 noise, t=1 data, x_t = t x_1 +
(1 - t) eps) rather than a VP diffusion model, so the pieces translate as:

* Tweedie estimate      x1_hat = x_t + (1 - t) v(t, x_t)
* Pi-GDM guidance       x1_hat' = x1_hat + lam * m * (y - x1_hat), then the step
                        uses the velocity implied by x1_hat' (a projection in the
                        clean domain, which is what Pi-GDM's pseudoinverse
                        guidance reduces to for a masking operator)
* SDEdit init           start at t0 from x_t0 = t0 y + (1 - t0) eps (DRUM's
                        t_init = 0.8 noise level is t0 = 0.2 here)
* RePaint harmonising   after each guided step, re-noise back and repeat
                        (`resample` extra passes per step)

Beyond DRUM, `mask_mode="ghost"` also leaves the prior free where it predicts a
return the sim does not have (sim pixel empty) or a return substantially nearer
than the sim surface. That is what lets ghost / multipath points appear, which
DRUM never needed because it only ever removes points.

All images are in the model's encoding: log-scaled depth in [-1, 1], empty = -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class GuidanceConfig:
    t0: float = 0.2  # SDEdit start time (DRUM t_init = 0.8 noise)
    num_steps: int = 32  # Euler steps over [t0, 1]
    lam: float = 1.0  # guidance strength in the clean domain (1 = hard replace)
    resample: int = 1  # RePaint passes per step (DRUM: 3)
    # Encoded-depth threshold separating a return from a drop. None = the
    # encoding of the model's min_depth (about -0.76 for 0.5..30 m), i.e. a pixel
    # is a return iff it decodes to a valid depth. Empty pixels are exactly -1.
    # DRUM's -0.3 is for a different range encoding and would not transfer.
    eta: float | None = None
    mask_mode: Literal["drum", "ghost"] = "ghost"
    # ghost mode: a predicted return nearer than (1 - ghost_rel) * sim depth is
    # treated as a ghost and left to the prior.
    ghost_rel: float = 0.25
    # Guidance is off for the last fraction of steps, so the final sample is the
    # prior's own (keeps realistic texture instead of a pasted sim image).
    stop_guidance_at: float = 1.0


def _metric(lidar_utils, x: torch.Tensor) -> torch.Tensor:
    return lidar_utils.restore_metric_depth(x.clamp(-1, 1))


def is_return(x: torch.Tensor, lidar_utils, cfg: GuidanceConfig) -> torch.Tensor:
    """Pixels that decode to a valid return (above eta and inside the depth window)."""
    valid = _metric(lidar_utils, x) > 0
    if cfg.eta is not None:
        valid = valid & (x > cfg.eta)
    return valid


def guidance_mask(
    x1_hat: torch.Tensor, y: torch.Tensor, lidar_utils, cfg: GuidanceConfig
) -> torch.Tensor:
    """1 where the sample is pulled toward the sim scan, 0 where the prior is free."""
    pred_valid = is_return(x1_hat, lidar_utils, cfg)
    if cfg.mask_mode == "drum":
        # Free only where the prior drops the ray. Where the sim is empty the
        # guidance pulls to -1, i.e. the sim's empties are kept.
        return pred_valid.to(x1_hat)
    sim_valid = is_return(y, lidar_utils, cfg)
    nearer = _metric(lidar_utils, x1_hat) < (1.0 - cfg.ghost_rel) * _metric(lidar_utils, y)
    keep = pred_valid & sim_valid & ~nearer
    return keep.to(x1_hat)


@torch.no_grad()
def realify(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    lidar_utils,
    cfg: GuidanceConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a real-looking scan close to the sim scan `y` (both encoded)."""
    def randn_like(x):
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    t0 = float(cfg.t0)
    x = t0 * y + (1.0 - t0) * randn_like(y)
    ts = torch.linspace(t0, 1.0, cfg.num_steps + 1, device=y.device)
    for k in range(cfg.num_steps):
        t, s = float(ts[k]), float(ts[k + 1])
        guided = t < cfg.stop_guidance_at
        for r in range(cfg.resample + 1):
            tt = torch.full((y.shape[0],), t, device=y.device, dtype=y.dtype)
            v = velocity(tt, x)
            x1_hat = x + (1.0 - t) * v
            if guided:
                m = guidance_mask(x1_hat, y, lidar_utils, cfg)
                x1_hat = x1_hat + cfg.lam * m * (y - x1_hat)
                v = (x1_hat - x) / (1.0 - t)
            x_next = x + (s - t) * v
            if r < cfg.resample and guided:
                # Re-noise s -> t under the linear interpolant: x_t = (t/s) x_s + std * eps,
                # std chosen so the marginal noise level is exactly (1 - t).
                std = ((1.0 - t) ** 2 - (t / s * (1.0 - s)) ** 2) ** 0.5
                x = (t / s) * x_next + std * randn_like(x_next)
            else:
                x = x_next
    return x


def compose(
    x_gen: torch.Tensor,
    y: torch.Tensor,
    lidar_utils,
    cfg: GuidanceConfig,
    mode: Literal["sim-range", "ghost", "full"] = "ghost",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the final pseudo-real metric depth from the generated sample.

    sim-range: DRUM exactly -- the sim range wherever both have a return, so the
               geometry is untouched and only raydrop is added.
    ghost:     as sim-range, plus generated returns where the sim has none or
               where the generated return is much nearer (ghost / multipath).
    full:      the generated range on every generated return (keeps range noise).

    Returns metric depth (0 = empty) and per-pixel masks for statistics.
    """
    gen_valid = is_return(x_gen, lidar_utils, cfg)
    sim_valid = is_return(y, lidar_utils, cfg)
    d_gen = _metric(lidar_utils, x_gen) * gen_valid
    d_sim = _metric(lidar_utils, y) * sim_valid
    nearer = gen_valid & sim_valid & (d_gen < (1.0 - cfg.ghost_rel) * d_sim)
    kept = gen_valid & sim_valid & ~nearer
    added = gen_valid & (~sim_valid | nearer)
    dropped = sim_valid & ~gen_valid

    if mode == "sim-range":
        depth = torch.where(gen_valid & sim_valid, d_sim, torch.zeros_like(d_sim))
    elif mode == "ghost":
        depth = torch.where(kept, d_sim, torch.zeros_like(d_sim))
        depth = torch.where(added, d_gen, depth)
    elif mode == "full":
        depth = d_gen
    else:
        raise ValueError(f"unknown compose mode {mode!r}")
    depth = depth * lidar_utils.get_mask(depth)
    return depth, {"kept": kept, "added": added, "dropped": dropped}

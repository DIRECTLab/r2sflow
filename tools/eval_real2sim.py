#!/usr/bin/env python3
"""Score a conditional real->sim model.

Paired (held-out sim episodes, so there is ground truth):
    pseudo-real test pair  --model-->  prediction  vs  the sim scan it came from.
    The identity baseline (prediction = the pseudo-real input) shows how much of
    the corruption the model removes.

Unpaired (the real bag's held-out tail, no ground truth):
    real scan --model--> prediction, compared as a distribution with sim test
    scans (fill fraction, per-row fill profile, depth histogram) next to the raw
    real scans, plus side-by-side figures.

    python tools/eval_real2sim.py --ckpt <cond .pth> --pairs <pair dir> \
        --steps 1 4 32 --out logs/eval_real2sim/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import r2flow.utils.flow as flow  # noqa: E402
from r2flow.utils.inference import setup_model  # noqa: E402
from r2flow.utils.training import PairedScans  # noqa: E402

GHOST_REL = 0.25  # a return this much nearer than the reference counts as a ghost


def load_hf(name: str, split: str, num_workers: int):
    import datasets as ds

    return ds.load_dataset(
        path=f"r2flow/data/{name}", name="spherical-64x1024", split=split,
        num_proc=num_workers, trust_remote_code=True,
    ).with_format("torch")


@torch.no_grad()
def translate(model, lidar_utils, cfg, depth_in, steps: int, generator, bf16: bool):
    """Real (or pseudo-real) metric depth -> predicted sim metric depth."""
    cond = flow.encode(lidar_utils, depth_in, None, cfg.data.data_format, False)
    noise = torch.randn(cond.shape, generator=generator, device=cond.device)
    x0 = cond + cfg.flow.bridge_sigma * noise if cfg.flow.source == "bridge" else noise

    def velocity(t, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return model(t, x, cond=cond).float()

    x1 = flow.euler_trajectory(velocity, x0, num_steps=steps)[-1]
    return flow.decode(lidar_utils, x1, cfg.data.data_format, False)[0]


def chamfer(xyz_a: torch.Tensor, xyz_b: torch.Tensor, max_points: int, gen) -> float:
    """Symmetric mean nearest-neighbour distance (m) between two point sets."""
    if len(xyz_a) == 0 or len(xyz_b) == 0:
        return float("nan")
    def sub(p):
        if len(p) <= max_points:
            return p
        idx = torch.randperm(len(p), generator=gen, device="cpu")[:max_points].to(p.device)
        return p[idx]
    a, b = sub(xyz_a), sub(xyz_b)
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean()).item()


def paired_metrics(pred, ref, lidar_utils, gen, max_points=8192):
    """Per-sample metrics of predicted vs reference metric depth (B,1,H,W)."""
    out = []
    xyz_p = lidar_utils.convert_metric_depth(pred, format="cartesian")
    xyz_r = lidar_utils.convert_metric_depth(ref, format="cartesian")
    for i in range(pred.shape[0]):
        p, r = pred[i, 0], ref[i, 0]
        vp, vr = p > 0, r > 0
        both = vp & vr
        rel = ((p - r).abs() / r.clamp(min=1e-6))[both]
        ghost = vp & (~vr | (p < (1 - GHOST_REL) * r))
        out.append({
            "abs_rel": rel.mean().item() if both.any() else float("nan"),
            "mae_m": (p - r).abs()[both].mean().item() if both.any() else float("nan"),
            "fill_err": abs(vp.float().mean().item() - vr.float().mean().item()),
            "ghost_rate": (ghost.float().sum() / vp.float().sum().clamp(min=1)).item(),
            "miss_rate": ((vr & ~vp).float().sum() / vr.float().sum().clamp(min=1)).item(),
            "chamfer_m": chamfer(
                xyz_p[i].flatten(1).T[vp.flatten()], xyz_r[i].flatten(1).T[vr.flatten()],
                max_points, gen,
            ),
        })
    return out


def mean_dict(rows):
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def distribution_stats(depths: torch.Tensor, max_depth: float) -> dict:
    """Summary of a set of metric depth images (N,1,H,W) for unpaired comparison."""
    valid = depths > 0
    fill = valid.float().mean(dim=(1, 2, 3))
    row = valid.float().mean(dim=(0, 1, 3))
    hist = torch.histc(depths[valid].float().log1p(), bins=64, min=0, max=float(np.log1p(max_depth)))
    near = (valid & (depths < 1.0)).float().sum() / valid.float().sum().clamp(min=1)
    return {"fill": fill.cpu().numpy(), "row_profile": row.cpu().numpy(),
            "hist": (hist / hist.sum().clamp(min=1)).cpu().numpy(), "near_1m_frac": near.item()}


def jsd(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def compare(a: dict, ref: dict) -> dict:
    return {
        "fill_mean": float(a["fill"].mean()),
        "fill_gap": float(abs(a["fill"].mean() - ref["fill"].mean())),
        "row_profile_l1": float(np.abs(a["row_profile"] - ref["row_profile"]).mean()),
        "depth_hist_jsd": jsd(a["hist"], ref["hist"]),
        "near_1m_frac": a["near_1m_frac"],
    }


def figure(path, panels, titles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.concatenate([p[p > 0] for p in panels if (p > 0).any()] or [np.ones(1)])
    clip = max(float(np.percentile(vals, 99)), 1.0)
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 1.4 * len(panels)), constrained_layout=True)
    for ax, img, title in zip(np.atleast_1d(axes), panels, titles):
        ax.imshow(np.ma.masked_where(img <= 0, np.minimum(img, clip)), cmap="turbo",
                  vmin=0, vmax=clip, aspect="auto", interpolation="nearest")
        ax.set_title(f"{title}  fill={(img > 0).mean():.3f}", fontsize=9)
        ax.axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True, help="pair dir (test split is used)")
    p.add_argument("--real", default="go2_common_real")
    p.add_argument("--sim", default="go2w_common_sim")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 4, 32])
    p.add_argument("--max-pairs", type=int, default=256)
    p.add_argument("--max-real", type=int, default=339)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--figures", type=int, default=12)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model, lidar_utils, cfg = setup_model(args.ckpt, device=device)
    assert model.cond_channels == 1, "expected a conditional (cond_channels=1) model"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    gen_cpu = torch.Generator().manual_seed(0)
    report = {"ckpt": str(args.ckpt), "pairs": str(args.pairs), "source": cfg.flow.source}

    # ---------------- paired: pseudo-real test pairs vs their sim ground truth
    pairs = PairedScans(args.pairs, split="test")
    idx = np.linspace(0, len(pairs) - 1, min(args.max_pairs, len(pairs))).astype(int)
    rows = {"identity": []} | {f"steps={s}": [] for s in args.steps}
    timing = {f"steps={s}": [] for s in args.steps}
    nfig = 0
    for start in range(0, len(idx), args.batch_size):
        items = [pairs[int(i)] for i in idx[start : start + args.batch_size]]
        ref = torch.stack([it["depth"] for it in items]).to(device)
        cond = torch.stack([it["cond_depth"] for it in items]).to(device)
        rows["identity"] += paired_metrics(cond, ref, lidar_utils, gen_cpu)
        preds = {}
        for s in args.steps:
            g = torch.Generator(device=device).manual_seed(start)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t = time.perf_counter()
            preds[s] = translate(model, lidar_utils, cfg, cond, s, g, args.bf16)
            torch.cuda.synchronize() if device.type == "cuda" else None
            timing[f"steps={s}"].append((time.perf_counter() - t) / len(items))
            rows[f"steps={s}"] += paired_metrics(preds[s], ref, lidar_utils, gen_cpu)
        for k in range(len(items)):
            if nfig >= args.figures:
                break
            panels = [cond[k, 0].cpu().numpy()] + [preds[s][k, 0].cpu().numpy() for s in args.steps]
            panels.append(ref[k, 0].cpu().numpy())
            figure(args.out / "figures" / f"pair_{items[k]['sample_id']:06d}.png", panels,
                   ["pseudo-real input"] + [f"pred ({s} steps)" for s in args.steps] + ["sim GT"])
            nfig += 1
    report["paired"] = {k: mean_dict(v) for k, v in rows.items()}
    report["sec_per_scan"] = {k: float(np.mean(v)) for k, v in timing.items()}

    # ---------------- unpaired: real held-out tail vs sim test distribution
    real = load_hf(args.real, "test", args.num_workers)
    sim = load_hf(args.sim, "test", args.num_workers)
    sim_idx = np.linspace(0, len(sim) - 1, min(512, len(sim))).astype(int).tolist()
    sim_depth = torch.cat([sim[sim_idx[i:i + 64]]["depth"] for i in range(0, len(sim_idx), 64)])
    ref_stats = distribution_stats(sim_depth, lidar_utils.max_depth)
    real_idx = list(range(min(args.max_real, len(real))))
    outs = {"real (raw)": []} | {f"steps={s}": [] for s in args.steps}
    nfig = 0
    for start in range(0, len(real_idx), args.batch_size):
        depth = real[real_idx[start : start + args.batch_size]]["depth"].to(device)
        outs["real (raw)"].append(depth.cpu())
        preds = {}
        for s in args.steps:
            g = torch.Generator(device=device).manual_seed(10_000 + start)
            preds[s] = translate(model, lidar_utils, cfg, depth, s, g, args.bf16)
            outs[f"steps={s}"].append(preds[s].cpu())
        for k in range(0, depth.shape[0], 8):
            if nfig >= args.figures:
                break
            panels = [depth[k, 0].cpu().numpy()] + [preds[s][k, 0].cpu().numpy() for s in args.steps]
            figure(args.out / "figures" / f"real_{real_idx[start + k]:05d}.png", panels,
                   ["real input"] + [f"real->sim ({s} steps)" for s in args.steps])
            nfig += 1
    report["unpaired_vs_sim"] = {
        "sim (reference)": compare(ref_stats, ref_stats),
        **{k: compare(distribution_stats(torch.cat(v), lidar_utils.max_depth), ref_stats)
           for k, v in outs.items()},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn hospital sim scans into pseudo-real scans with a real-data flow prior.

DRUM-style (see r2flow/utils/guidance.py): each sim range image is noised to t0
and integrated through the real prior with raydrop/ghost-aware guidance toward
the sim scan. Every compose mode is written from the same sample, one pair
directory each, ready for `train.py --dataset go2w_pseudoreal --data_dir ...`:

    <out>/<compose>/<sim_id>_s<seed>.npz     depth_sim, depth_pr  (1, H, W) metric
    <out>/<compose>/frames.json              written by --merge (after all shards)
    <out>/stats_shard<i>.json                kept / added / dropped fractions
    <out>/figures/                           side-by-side PNGs for --figures frames

Shard across GPUs with --shard i --num-shards n (e.g. a Slurm array), then run
once more with --merge.

    python tools/make_pseudoreal.py --ckpt <real prior .pth> --out data/pr --seeds 4
    python tools/make_pseudoreal.py --out data/pr --merge
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import r2flow.utils.flow as flow  # noqa: E402
from r2flow.utils.guidance import GuidanceConfig, compose, realify  # noqa: E402
from r2flow.utils.inference import setup_model  # noqa: E402

COMPOSE_MODES = ("sim-range", "ghost", "full")
# frames.json holding each sim sample's split, per sim dataset.
SIM_FRAMES = {
    "go2w_common_sim": Path("r2flow/data/go2w_nav_sim/dataset/frames.json"),
    "go2w_l2_sim": Path("r2flow/data/go2w_l2_sim/dataset/frames.json"),
}


def load_sim(split: str, num_workers: int, sim_dataset: str = "go2w_common_sim"):
    import datasets as ds

    return ds.load_dataset(
        path=f"r2flow/data/{sim_dataset}",
        name="spherical-64x1024",
        split=split,
        num_proc=num_workers,
        trust_remote_code=True,
    ).with_format("torch")


def save_figure(path: Path, d_sim, d_gen, d_out, max_depth: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip = max(float(np.percentile(d_sim[d_sim > 0], 99)) if (d_sim > 0).any() else 1.0, 1.0)
    fig, axes = plt.subplots(3, 1, figsize=(16, 4.2), constrained_layout=True)
    for ax, img, title in zip(
        axes, (d_sim, d_gen, d_out), ("sim", "prior sample", "pseudo-real (composed)")
    ):
        show = np.ma.masked_where(img <= 0, np.minimum(img, clip))
        ax.imshow(show, cmap="turbo", vmin=0, vmax=clip, aspect="auto", interpolation="nearest")
        ax.set_title(f"{title}  fill={float((img > 0).mean()):.3f}", fontsize=9)
        ax.axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def merge(out: Path, sim_dataset: str = "go2w_common_sim") -> None:
    split_of = {f["sample_id"]: f["split"] for f in json.loads(SIM_FRAMES[sim_dataset].read_text())}
    for mode in COMPOSE_MODES:
        d = out / mode
        if not d.exists():
            continue
        frames = []
        for path in sorted(d.glob("*.npz")):
            sim_id, seed = path.stem.split("_s")
            frames.append({
                "file_name": path.name,
                "sim_sample_id": int(sim_id),
                "seed": int(seed),
                "split": split_of[int(sim_id)],
            })
        (d / "frames.json").write_text(json.dumps(frames, indent=1))
        n_train = sum(f["split"] == "train" for f in frames)
        print(f"{mode}: {len(frames)} pairs ({n_train} train, {len(frames) - n_train} test)")

    stats = [json.loads(p.read_text()) for p in sorted(out.glob("stats_shard*.json"))]
    if stats:
        keys = [k for k in stats[0] if k.endswith("_frac") or k.endswith("_fill")]
        total = sum(s["num_samples"] for s in stats)
        summary = {k: sum(s[k] * s["num_samples"] for s in stats) / total for k in keys}
        summary["num_samples"] = total
        summary["guidance"] = stats[0]["guidance"]
        (out / "stats.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


@torch.no_grad()
def generate(args) -> None:
    device = torch.device(args.device)
    model, lidar_utils, cfg = setup_model(args.ckpt, device=device)
    assert model.cond_channels == 0, "the prior must be an unconditional model"
    assert not cfg.data.train_reflectance, "expected a depth-only prior"
    gcfg = GuidanceConfig(
        t0=args.t0, num_steps=args.steps, lam=args.lam, resample=args.resample,
        eta=args.eta, mask_mode=args.mask_mode, ghost_rel=args.ghost_rel,
        stop_guidance_at=args.stop_guidance_at,
    )
    print(gcfg)

    def velocity(t, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            return model(t, x).float()

    out = Path(args.out)
    for mode in COMPOSE_MODES:
        (out / mode).mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    totals = {"kept_frac": 0.0, "added_frac": 0.0, "dropped_frac": 0.0,
              "sim_fill": 0.0, "pr_fill": 0.0, "num_samples": 0}
    num_figs = 0
    for split in args.splits:
        data = load_sim(split, args.num_workers, args.sim_dataset)
        ids = [i for i in range(len(data)) if i % args.num_shards == args.shard]
        if args.limit is not None:
            ids = ids[: args.limit]
        print(f"{split}: {len(ids)} sim frames on shard {args.shard}/{args.num_shards}")
        for start in range(0, len(ids), args.batch_size):
            batch = data[ids[start : start + args.batch_size]]
            depth = batch["depth"].to(device)
            sample_ids = batch["sample_id"].tolist()
            y = flow.encode(lidar_utils, depth, None, cfg.data.data_format, False)
            for seed in range(args.seeds):
                todo = [
                    j for j, sid in enumerate(sample_ids)
                    if args.overwrite
                    or not (out / "ghost" / f"{sid:06d}_s{seed}.npz").exists()
                ]
                if not todo:
                    continue
                gen = torch.Generator(device=device).manual_seed(
                    args.base_seed + 1000003 * seed + sample_ids[0]
                )
                x = realify(velocity, y[todo], lidar_utils, gcfg, generator=gen).clamp(-1, 1)
                d_gen = flow.decode(lidar_utils, x, cfg.data.data_format, False)[0]
                d_sim = depth[todo]
                composed = {}
                for mode in COMPOSE_MODES:
                    composed[mode], masks = compose(x, y[todo], lidar_utils, gcfg, mode)
                n = len(todo)
                pix = masks["kept"][0].numel()
                totals["kept_frac"] += masks["kept"].float().sum().item() / pix
                totals["added_frac"] += masks["added"].float().sum().item() / pix
                totals["dropped_frac"] += masks["dropped"].float().sum().item() / pix
                totals["sim_fill"] += (d_sim > 0).float().sum().item() / pix
                totals["pr_fill"] += (composed["ghost"] > 0).float().sum().item() / pix
                totals["num_samples"] += n
                for k, j in enumerate(todo):
                    sid = sample_ids[j]
                    for mode in COMPOSE_MODES:
                        np.savez_compressed(
                            out / mode / f"{sid:06d}_s{seed}.npz",
                            depth_sim=d_sim[k].cpu().numpy().astype(np.float32),
                            depth_pr=composed[mode][k].cpu().numpy().astype(np.float32),
                        )
                    if num_figs < args.figures and seed == 0:
                        save_figure(
                            fig_dir / f"{split}_{sid:06d}.png",
                            d_sim[k, 0].cpu().numpy(), d_gen[k, 0].cpu().numpy(),
                            composed["ghost"][k, 0].cpu().numpy(), lidar_utils.max_depth,
                        )
                        num_figs += 1
            print(f"  {split} {min(start + args.batch_size, len(ids))}/{len(ids)}", flush=True)

    n = max(totals["num_samples"], 1)
    stats = {k: (v / n if k != "num_samples" else v) for k, v in totals.items()}
    stats["guidance"] = dataclasses.asdict(gcfg)
    stats["ckpt"] = str(args.ckpt)
    (out / f"stats_shard{args.shard}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, help="unconditional real prior checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merge", action="store_true", help="only write frames.json/stats.json")
    p.add_argument("--sim-dataset", default="go2w_common_sim", choices=sorted(SIM_FRAMES),
                   help="sim scans to realify (go2w_l2_sim: the L2-emulating sim)")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--seeds", type=int, default=4, help="pseudo-real draws per sim scan")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="sim frames per split (smoke tests)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--figures", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g = p.add_argument_group("guidance (r2flow/utils/guidance.py)")
    d = GuidanceConfig()
    g.add_argument("--t0", type=float, default=d.t0)
    g.add_argument("--steps", type=int, default=d.num_steps)
    g.add_argument("--lam", type=float, default=d.lam)
    g.add_argument("--resample", type=int, default=d.resample)
    g.add_argument("--eta", type=float, default=d.eta)
    g.add_argument("--mask-mode", choices=("drum", "ghost"), default=d.mask_mode)
    g.add_argument("--ghost-rel", type=float, default=d.ghost_rel)
    g.add_argument("--stop-guidance-at", type=float, default=d.stop_guidance_at)
    args = p.parse_args(argv)
    if args.merge:
        merge(args.out, args.sim_dataset)
    else:
        assert args.ckpt is not None, "--ckpt is required to generate"
        generate(args)


if __name__ == "__main__":
    main()

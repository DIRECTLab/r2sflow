#!/usr/bin/env python3
"""Evaluate what the flow does to sim and to real scans, against a sim reference.

Sibling to evaluate.py rather than an extension of it: that script's contract is
"a directory of generated .pth samples vs one hardcoded dataset", and rewiring it
would put the paper-reproduction path at risk.

Arms, all scored against the same REFERENCE:

    NULL   reference split against itself      -- the floor; scores are read
                                                  against this, not against zero
    A      sim seed  -> noise to t0 -> flow    -- in-distribution
    A'     as A, but fill matched to real      -- isolates the density confound
    B      real seed -> noise to t0 -> flow    -- the transfer case

Sim scans are ~80% filled and real ones ~39%, with no overlap, so any A-vs-B gap
is explainable by point density alone. A' is what makes the gap interpretable.

Run inside the container.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r2flow.utils
from r2flow.metrics import features as feat


def parse_range(spec: str | None, total: int):
    """'0:118' -> list(range(0, 118)); None -> everything."""
    if spec is None:
        return None
    start, _, end = spec.partition(":")
    return list(range(int(start or 0), int(end or total)))


def sparsify(depth, mask, target_fill: float, generator=None):
    """Randomly drop valid pixels until the fill fraction matches `target_fill`."""
    current = mask.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    keep_rate = (target_fill / current).clamp(max=1.0)
    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) < keep_rate).float()
    new_mask = mask * keep
    return depth * new_mask, new_mask


@torch.no_grad()
def extract_source(proj, loader, scales, device, metrics, tpred, lidar_utils, cfg):
    """REFERENCE path: scans straight through, no model."""
    chunks = []
    offset = 0
    for batch in tqdm(loader, desc="reference", leave=False):
        n = batch["depth"].shape[0]
        model_input = None
        if tpred is not None:
            model_input = r2flow.utils.flow.encode(
                lidar_utils,
                batch["depth"].to(device),
                batch["reflectance"].to(device),
                cfg.data.data_format,
                cfg.data.train_reflectance,
            )
        per_metric, hists, _ = feat.extract_batch(
            proj, batch, scales, device, metrics, tpred=tpred, model_input=model_input
        )
        chunks.append((per_metric, hists, np.arange(offset, offset + n)))
        offset += n
    return feat.accumulate(chunks)


@torch.no_grad()
def run_arm(
    model, lidar_utils, cfg, proj, loader, scales, device, metrics, tpred,
    t0: float, repeats: int, dt: float, target_fill: float | None, label: str,
):
    """Transport each seed scan `repeats` times and extract features from the output.

    Noise is seeded by draw index only, so arm A and arm B see the same noise
    tensors at the same (scan, draw) -- the comparison is paired rather than two
    independent Monte Carlo runs.
    """
    flow_matcher = r2flow.utils.flow.make_flow_matcher(cfg)
    shape = (
        r2flow.utils.flow.num_channels(cfg.data.data_format, cfg.data.train_reflectance),
        *cfg.data.resolution,
    )
    num_steps = max(1, int(round((1.0 - t0) / dt)))
    chunks = []
    offset = 0
    fills = []

    for batch in tqdm(loader, desc=f"{label} t0={t0:g}", leave=False):
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)
        reflectance = batch["reflectance"].to(device)
        n = depth.shape[0]

        if target_fill is not None:
            generator = torch.Generator(device=device).manual_seed(1234 + offset)
            depth, mask = sparsify(depth, mask, target_fill, generator)
        fills.append(mask.mean().item())

        x_1 = r2flow.utils.flow.encode(
            lidar_utils, depth, reflectance, cfg.data.data_format, cfg.data.train_reflectance
        )

        for draw in range(repeats):
            seeds = torch.full((n,), draw, dtype=torch.long)
            x_0 = r2flow.utils.training.restore_x_0(seeds, shape, device)
            x_t = r2flow.utils.flow.noise_to_t0(flow_matcher, x_1, x_0, t0)
            if t0 >= 1.0:
                x_hat = x_t  # identity; the model is not in the loop
            else:
                x_hat = r2flow.utils.flow.euler_trajectory(
                    model, x_t, num_steps=num_steps, t_start=t0, t_end=1.0
                )[-1]

            out_depth, out_xyz, out_rflct = r2flow.utils.flow.decode(
                lidar_utils, x_hat, cfg.data.data_format, cfg.data.train_reflectance
            )
            out_mask = lidar_utils.get_mask(out_depth)
            sample = {
                "depth": out_depth, "xyz": out_xyz,
                "reflectance": out_rflct if out_rflct.numel() else torch.zeros_like(out_depth),
                "mask": out_mask,
            }
            per_metric, hists, _ = feat.extract_batch(
                proj, sample, scales, device, metrics, tpred=tpred, model_input=x_hat
            )
            chunks.append((per_metric, hists, np.arange(offset, offset + n)))
        offset += n

    result = feat.accumulate(chunks)
    result.info = {"t0": t0, "repeats": repeats, "num_steps": num_steps,
                   "input_fill": float(np.mean(fills))}
    return result


@torch.no_grad()
def run_ladder(proj, loader, scales, device, metrics, tpred, lidar_utils, cfg,
               kind: str, levels):
    """Score parametrically corrupted scans, to check the metric is monotone.

    Goes through the same encode -> decode path the arms do, so a failure here is
    about the metric rather than about the transport. `noise` perturbs in the
    normalised space the model works in; `dropout` removes valid returns, which is
    the axis the sim/real fill gap lives on.
    """
    out = {}
    for level in levels:
        chunks, offset = [], 0
        for batch in loader:
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            reflectance = batch["reflectance"].to(device)
            n = depth.shape[0]
            generator = torch.Generator(device=device).manual_seed(99 + offset)

            if kind == "dropout":
                keep = (torch.rand(mask.shape, device=device, generator=generator) >= level).float()
                depth, mask = depth * mask * keep, mask * keep

            x = r2flow.utils.flow.encode(
                lidar_utils, depth, reflectance, cfg.data.data_format,
                cfg.data.train_reflectance
            )
            if kind == "noise":
                x = x + level * torch.randn(x.shape, device=device, generator=generator)

            out_depth, out_xyz, out_rflct = r2flow.utils.flow.decode(
                lidar_utils, x, cfg.data.data_format, cfg.data.train_reflectance
            )
            sample = {
                "depth": out_depth, "xyz": out_xyz,
                "reflectance": out_rflct if out_rflct.numel() else torch.zeros_like(out_depth),
                "mask": lidar_utils.get_mask(out_depth),
            }
            per_metric, hists, _ = feat.extract_batch(
                proj, sample, scales, device, metrics, tpred=tpred, model_input=x
            )
            chunks.append((per_metric, hists, np.arange(offset, offset + n)))
            offset += n
        out[level] = feat.accumulate(chunks)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None,
                   help="defaults to r2flow/data/<cfg.data.dataset>")
    p.add_argument("--reference-split", default="train")
    p.add_argument("--reference-frames", default="0:118",
                   help="index range within the split; a guard band before the "
                        "Arm A seeds keeps the two temporally disjoint")
    p.add_argument("--arm-a", type=Path, default=None, help="defaults to --reference")
    p.add_argument("--arm-a-split", default="test")
    p.add_argument("--arm-b", type=Path, default=None)
    p.add_argument("--arm-b-split", default="test")
    p.add_argument("--arm-b-frames", default=None)
    p.add_argument("--t", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.8, 1.0])
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--arm-seeds", type=int, default=32,
                   help="cap every arm to this many seed scans, evenly spaced")
    p.add_argument("--dt", type=float, default=1.0 / 256,
                   help="Euler step SIZE (paper uses the euler sampler); fixed size "
                        "rather than fixed count so dt is not confounded with t0")
    p.add_argument("--t-predictor", type=Path, default=None)
    p.add_argument("--max-tpred-mae", type=float, default=0.15)
    p.add_argument("--metrics", nargs="+", default=["FRID", "FPD", "FSVD", "FPVD"])
    p.add_argument("--kid-dim", type=int, default=32)
    p.add_argument("--fd-dim", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rangenet-stats", choices=("auto", "kitti"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null-only", action="store_true",
                   help="stop after the reference and NULL floor (verification step 1)")
    p.add_argument("--ladder", action="store_true",
                   help="also run corruption ladders and report Spearman monotonicity")
    p.add_argument("--export", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    model, lidar_utils, cfg = r2flow.utils.inference.setup_model(args.ckpt, device=device)
    lidar_utils.to(device)

    scales = feat.MetricScales.from_cfg(cfg)
    metrics = list(args.metrics)
    skipped = {}
    if not cfg.data.train_reflectance and "FRD" in metrics:
        # FRD is the only metric that consumes reflectance. Dropping it for every
        # arm including the reference keeps them measured on the same footing;
        # computing it on one side only would produce a score with no counterpart.
        metrics.remove("FRD")
        skipped["FRD"] = "checkpoint trained with train_reflectance=False"

    reference_path = args.reference or Path(f"r2flow/data/{cfg.data.dataset}")
    arm_a_path = args.arm_a or reference_path

    print(f"checkpoint : {args.ckpt}")
    print(f"resolution : {tuple(cfg.data.resolution)}  format={cfg.data.data_format}")
    print(f"scales     : depth[{scales.min_depth}, {scales.max_depth}] "
          f"bev field={scales.bev_field_size} bins={scales.bev_bins}")
    print(f"metrics    : {metrics}" + (f"  skipped={skipped}" if skipped else ""))

    tpred_model, tpred_cfg = None, {}
    if args.t_predictor is not None:
        from r2flow.metrics import tpred as tpred_mod

        tpred_model, tpred_cfg = tpred_mod.load(args.t_predictor, device=device)
        mae = tpred_cfg.get("val_mae_t", float("nan"))
        print(f"t-predictor: {args.t_predictor}  mae_t={mae:.4f}")
        if not np.isfinite(mae) or mae > args.max_tpred_mae:
            raise SystemExit(
                f"t-predictor mae_t={mae} exceeds --max-tpred-mae {args.max_tpred_mae}; "
                "a degenerate predictor yields near-constant features and an FTD of "
                "~0 for everything, which would look like a passing result."
            )
        metrics.append("FTD")

    def make_loader(path, split, frames, shuffle=False, cap=None):
        scans = feat.open_scans(path, cfg.data.resolution, split=split,
                                select=frames, num_workers=args.num_workers)
        if cap is not None and len(scans) > cap:
            # Every arm must contribute the same number of scans: FD's dependence
            # on n is strong enough that unequal counts would reorder the arms on
            # their own. Evenly spaced rather than the first N, so the seeds still
            # span the whole trajectory.
            picked = np.linspace(0, len(scans) - 1, cap).round().astype(int).tolist()
            scans = feat.open_scans(path, cfg.data.resolution, split=split,
                                    select=picked if frames is None
                                    else [frames[i] for i in picked],
                                    num_workers=args.num_workers)
        return DataLoader(scans, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers), len(scans)

    # ---------------------------------------------------------------- reference
    ref_loader, n_ref = make_loader(
        reference_path, args.reference_split, parse_range(args.reference_frames, 10**6)
    )
    proj = feat.build_extractor(cfg.data.resolution, scales,
                                [m for m in metrics if m != "FTD"])
    proj.to(device)
    print(f"reference  : {reference_path} split={args.reference_split} n={n_ref}")

    reference = extract_source(proj, ref_loader, scales, device, metrics,
                               tpred_model, lidar_utils, cfg)

    results = {
        "info": {
            "checkpoint": str(args.ckpt),
            "reference": str(reference_path),
            "reference_split": args.reference_split,
            "reference_frames": args.reference_frames,
            "n_reference_scans": reference.num_scans(),
            "metrics": metrics,
            "metrics_skipped": skipped,
            "scales": scales.__dict__,
            "t_predictor": str(args.t_predictor) if args.t_predictor else None,
            "t_predictor_cfg": {k: v for k, v in tpred_cfg.items() if k != "weights"},
            "repeats": args.repeats,
            "dt": args.dt,
            "sampler": "euler (fixed step size)",
            "guidance": (
                "KID is primary: it estimates no covariance and carries no "
                "sample-size bias. FD is reported at low dimension beside the NULL "
                "floor. Read every arm against NULL, not against zero."
            ),
        },
        "arms": {},
    }

    # One projection, fitted on the full reference, shared by NULL and every arm.
    projections = feat.fit_projections(reference, metrics, args.kid_dim, args.fd_dim)

    # --------------------------------------------------------------------- NULL
    # Three floors, because the reference is one continuous trajectory and a
    # reference-vs-reference score is not zero. `random` is the plumbing check;
    # `contiguous` is the floor an arm seeded from a disjoint later block should
    # actually be read against.
    null_descriptions = {
        "random": "reference vs itself, random scan split (pure sampling noise)",
        "interleaved": "reference vs itself, alternating blocks (mild temporal structure)",
        "contiguous": "reference vs itself, first half vs second half (includes scene drift)",
    }
    print("\nNULL floors (KID):")
    for mode, description in null_descriptions.items():
        first, second = feat.split_halves(reference, block=args.block, mode=mode,
                                          seed=args.seed)
        scores = feat.compare(first, second, metrics, projections,
                              bootstrap=args.bootstrap, block=args.block, seed=args.seed)
        results["arms"][f"NULL_{mode}"] = {
            "description": description,
            "n_scans": [first.num_scans(), second.num_scans()],
            "scores": scores,
        }
        summary = {m: round(v["KID"], 4) for m, v in scores.items()
                   if isinstance(v, dict) and "KID" in v}
        print(f"  {mode:12s} {summary}")

    # Participation ratio of every extractor on the reference. Judged relatively,
    # not against an absolute band: this reference is one trajectory in one room
    # and the pretrained SemanticKITTI extractors -- which cannot have collapsed
    # onto a flow timestep, having never seen one -- score 1.4 to 9.4 here. A low
    # absolute PR is therefore a property of the data. What would signal a
    # collapsed FTD is being far below the pretrained ones on the same scans.
    ratios = {m: feat.participation_ratio(v) for m, v in reference.feats.items()}
    results["info"]["participation_ratio"] = ratios
    print("participation ratio: " + "  ".join(f"{m}={v:.1f}" for m, v in ratios.items()))
    if "FTD" in ratios and len(ratios) > 1:
        pretrained = [v for m, v in ratios.items() if m != "FTD"]
        if ratios["FTD"] < 0.5 * min(pretrained):
            print(f"  WARNING: FTD ({ratios['FTD']:.1f}) is far below every pretrained "
                  f"extractor (min {min(pretrained):.1f}) -- likely collapsed onto the "
                  f"noise-level axis rather than encoding scene content.")

    # ------------------------------------------------------------------ ladders
    if args.ladder:
        from scipy.stats import spearmanr

        ladder_loader, _ = make_loader(arm_a_path, args.arm_a_split, None)
        ladders = {
            "noise": [0.01, 0.02, 0.05, 0.10, 0.20],
            "dropout": [0.05, 0.10, 0.20, 0.40, 0.80],
        }
        results["ladders"] = {}
        print("\ncorruption ladders (KID; want strictly increasing):")
        for kind, levels in ladders.items():
            sets = run_ladder(proj, ladder_loader, scales, device, metrics,
                              tpred_model, lidar_utils, cfg, kind, levels)
            per_level = {
                f"{lvl:g}": feat.compare(reference, fs, metrics, projections,
                                         bootstrap=0, seed=args.seed)
                for lvl, fs in sets.items()
            }
            results["ladders"][kind] = {"levels": levels, "scores": per_level}
            print(f"  {kind}:")
            for metric in metrics:
                series = [per_level[f"{lvl:g}"].get(metric, {}).get("KID") for lvl in levels]
                if any(v is None for v in series):
                    continue
                rho = spearmanr(levels, series).statistic
                flag = "OK " if rho > 0.99 else "NOT MONOTONE"
                print(f"    {metric:5s} rho={rho:+.3f} {flag} "
                      + " ".join(f"{v:+.4f}" for v in series))
                results["ladders"][kind].setdefault("spearman", {})[metric] = float(rho)

    if args.null_only:
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(results, indent=1, default=str))
            print(f"\nwrote {args.export}")
        return results

    # --------------------------------------------------------------------- arms
    arm_specs = [("A", arm_a_path, args.arm_a_split, None, None)]
    real_fill = None
    if args.arm_b is not None:
        arm_specs.append(("B", args.arm_b, args.arm_b_split,
                          parse_range(args.arm_b_frames, 10**6), None))

    # Measure real fill first so A' can be matched to it.
    if args.arm_b is not None:
        b_loader, _ = make_loader(args.arm_b, args.arm_b_split,
                                  parse_range(args.arm_b_frames, 10**6),
                                  cap=args.arm_seeds)
        fills = [batch["mask"].mean().item() for batch in b_loader]
        real_fill = float(np.mean(fills))
        print(f"arm B mean input fill: {real_fill:.4f}")
        arm_specs.insert(1, ("A'", arm_a_path, args.arm_a_split, None, real_fill))

    for label, path, split, frames, target_fill in arm_specs:
        loader, n = make_loader(path, split, frames, cap=args.arm_seeds)
        for t0 in args.t:
            arm = run_arm(model, lidar_utils, cfg, proj, loader, scales, device,
                          metrics, tpred_model, t0, args.repeats, args.dt,
                          target_fill, label)
            scores = feat.compare(reference, arm, metrics, projections,
                                  bootstrap=args.bootstrap, block=args.block,
                                  seed=args.seed)
            results["arms"].setdefault(label, {"description": str(path), "n_scans": n,
                                               "by_t0": {}})
            # Checksum so the t0=0 leak check below can compare arms exactly.
            checksum = {k: float(np.abs(v).sum()) for k, v in sorted(arm.feats.items())}
            results["arms"][label]["by_t0"][f"{t0:g}"] = {
                "info": arm.info, "scores": scores, "feature_checksum": checksum
            }
            summary = {m: round(v["KID"], 5) for m, v in scores.items()
                       if isinstance(v, dict) and "KID" in v}
            print(f"  {label:2s} t0={t0:<4g} fill={arm.info['input_fill']:.3f} "
                  f"steps={arm.info['num_steps']:<4d} KID={summary}")

    # --------------------------------------------------- t0=0 seed-leak check
    # At t0=0 the forward process returns the noise untouched, so the seed scan is
    # fully discarded and every arm is literally the same distribution. With common
    # random numbers across arms the features must be bit-identical; any difference
    # means the seed is reaching the output through some path other than x_t0.
    zero = [lab for lab in ("A", "A'", "B")
            if "0" in results["arms"].get(lab, {}).get("by_t0", {})]
    if len(zero) > 1:
        baseline = results["arms"][zero[0]]["by_t0"]["0"]["feature_checksum"]
        drift = {
            lab: max(abs(results["arms"][lab]["by_t0"]["0"]["feature_checksum"][k] - v)
                     for k, v in baseline.items())
            for lab in zero[1:]
        }
        results["info"]["t0_zero_leak_check"] = {
            "arms": zero, "max_checksum_drift": drift,
            "pass": all(v == 0.0 for v in drift.values()),
        }
        verdict = "PASS (identical)" if all(v == 0.0 for v in drift.values()) else "FAIL"
        print(f"\nt0=0 seed-leak check across {zero}: {verdict}  drift={drift}")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(results, indent=1, default=str))
        print(f"\nwrote {args.export}")
    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the timestep-predictor feature extractor used by evaluate_flow.py.

The pretext task is recovering the flow timestep `t` from `x_t`, paired with a
denoising head so the encoder keeps scene content rather than collapsing onto the
noise-level axis (see r2flow/metrics/tpred.py for why that matters).

The forward process comes from the same `torchcfm` flow matcher train.py uses,
constructed from the flow checkpoint's own config, so "the training forward
process" means the same thing here as it does during training.

Run inside the container:

    python tools/train_t_predictor.py \
        --ckpt logs/r2flow-1rf/go2w_sim_l1/spherical-64x512/.../checkpoint_0002560000.pth \
        --dataset r2flow/data/go2w_sim_l1 --split all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import r2flow.utils
from r2flow.metrics import features as feat
from r2flow.metrics import tpred


def augment_depth(depth, mask, dropout_max: float, scale_range: tuple[float, float]):
    """Augment metric depth before encoding.

    Valid-pixel dropout is the important one. Sim scans are ~80% filled and real
    ones ~39%, with no overlap, and an empty pixel is exactly -1.0 after encoding.
    Without this the encoder would learn to read fill fraction, and every arm
    comparison would reduce to counting valid pixels.
    """
    batch = depth.shape[0]
    device = depth.device

    scale = torch.empty(batch, 1, 1, 1, device=device).uniform_(*scale_range)
    depth = depth * scale

    keep_rate = 1.0 - torch.rand(batch, 1, 1, 1, device=device) * dropout_max
    keep = (torch.rand_like(depth) < keep_rate).float()
    mask = mask * keep
    return depth * mask, mask


def sample_t(batch: int, device, clean_fraction: float = 0.15):
    """t ~ (1-c)*U(0,1) + c*delta(1).

    Every image the extractor sees at evaluation time is at t=1 (clean reference
    scans and flow outputs integrated to 1). Under plain U(0,1) the top bin gets
    ~3% of the mass, leaving the clean end the least well-trained region of the
    feature space -- exactly where the metric operates.
    """
    t = torch.rand(batch, device=device)
    clean = torch.rand(batch, device=device) < clean_fraction
    return torch.where(clean, torch.ones_like(t), t)


def encode_batch(batch, lidar_utils, cfg, args, device, augment: bool):
    depth = batch["depth"].to(device)
    mask = batch["mask"].to(device)
    reflectance = batch["reflectance"].to(device)
    if augment:
        depth, mask = augment_depth(depth, mask, args.dropout_max, (args.scale_min, args.scale_max))
    x_1 = r2flow.utils.flow.encode(
        lidar_utils, depth, reflectance, cfg.data.data_format, cfg.data.train_reflectance
    )
    if augment:
        shift = int(torch.randint(0, x_1.shape[-1], (1,)).item())
        x_1 = x_1.roll(shift, dims=-1)
        if torch.rand(1).item() < 0.5:
            x_1 = x_1.flip(-1)
    return x_1


def evaluate_mae(model, loader, lidar_utils, cfg, args, device, flow_matcher, rounds=4):
    """Mean absolute error of predicted t, plus denoise L1."""
    model.eval()
    abs_errors, denoise = [], []
    with torch.no_grad():
        for _ in range(rounds):
            for batch in loader:
                x_1 = encode_batch(batch, lidar_utils, cfg, args, device, augment=False)
                t = sample_t(x_1.shape[0], device)
                _, x_t, _ = flow_matcher.sample_location_and_conditional_flow(
                    x0=torch.randn_like(x_1), x1=x_1, t=t
                )
                out = model(x_t)
                abs_errors.append((out["t"] - t).abs().mean().item())
                denoise.append(F.l1_loss(out["denoised"], x_1).item())
    model.train()
    return float(sum(abs_errors) / len(abs_errors)), float(sum(denoise) / len(denoise))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True,
                   help="flow checkpoint; supplies the resolution and encoding contract")
    p.add_argument("--dataset", type=Path, default=None,
                   help="defaults to r2flow/data/<cfg.data.dataset>")
    p.add_argument("--split", default="all")
    p.add_argument("--probe-dataset", type=Path, default=None,
                   help="optional out-of-domain set for a generalisation probe")
    p.add_argument("--backbone", default="resnet10t")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--dropout-max", type=float, default=0.6)
    p.add_argument("--scale-min", type=float, default=0.8)
    p.add_argument("--scale-max", type=float, default=1.25)
    p.add_argument("--w-bins", type=float, default=0.5)
    p.add_argument("--w-reg", type=float, default=0.2)
    p.add_argument("--w-denoise", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Only the preprocessing contract is wanted here; the flow weights are discarded.
    _, lidar_utils, cfg = r2flow.utils.inference.setup_model(args.ckpt, device=device)
    lidar_utils.to(device)
    flow_matcher = r2flow.utils.flow.make_flow_matcher(cfg)

    dataset_path = args.dataset or Path(f"r2flow/data/{cfg.data.dataset}")
    scans = feat.open_scans(dataset_path, cfg.data.resolution, split=args.split,
                            num_workers=args.num_workers)
    loader = DataLoader(scans, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    in_channels = r2flow.utils.flow.num_channels(
        cfg.data.data_format, cfg.data.train_reflectance
    )
    print(f"dataset: {dataset_path} split={args.split} n={len(scans)}")
    print(f"resolution: {tuple(cfg.data.resolution)}  channels: {in_channels}")

    model = tpred.TimestepPredictor(
        backbone=args.backbone,
        in_channels=in_channels,
        pretrained=args.pretrained,
        feature_dim=args.feature_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone: {args.backbone}  params: {n_params:,}  feat_dim: {model.num_channels}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = r2flow.utils.training.get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.steps
    )

    step, started = 0, time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            x_1 = encode_batch(batch, lidar_utils, cfg, args, device, augment=True)
            t = sample_t(x_1.shape[0], device)
            _, x_t, _ = flow_matcher.sample_location_and_conditional_flow(
                x0=torch.randn_like(x_1), x1=x_1, t=t
            )
            out = model(x_t)

            target = tpred.soft_bin_targets(t, model.NUM_BINS)
            loss_bins = -(target * torch.log_softmax(out["bins"], dim=-1)).sum(-1).mean()
            loss_reg = F.l1_loss(out["t"], t)
            loss_denoise = F.l1_loss(out["denoised"], x_1)
            loss = args.w_bins * loss_bins + args.w_reg * loss_reg + args.w_denoise * loss_denoise

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == args.steps:
                mae = (out["t"] - t).abs().mean().item()
                print(f"  step {step:6d}/{args.steps}  loss={loss.item():.4f}  "
                      f"bins={loss_bins.item():.4f} reg={loss_reg.item():.4f} "
                      f"denoise={loss_denoise.item():.4f}  mae_t={mae:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-started:.0f}s")

    mae_t, denoise_l1 = evaluate_mae(model, loader, lidar_utils, cfg, args, device, flow_matcher)
    print(f"\nin-sample mae_t={mae_t:.4f} (chance 0.25)  denoise_l1={denoise_l1:.4f}")

    probe = {}
    if args.probe_dataset is not None:
        probe_scans = feat.open_scans(args.probe_dataset, cfg.data.resolution, split="all",
                                      num_workers=args.num_workers)
        probe_loader = DataLoader(probe_scans, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        p_mae, p_den = evaluate_mae(model, probe_loader, lidar_utils, cfg, args, device,
                                    flow_matcher, rounds=1)
        probe = {"probe_mae_t": p_mae, "probe_denoise_l1": p_den}
        print(f"out-of-domain probe ({args.probe_dataset.name}): "
              f"mae_t={p_mae:.4f} denoise_l1={p_den:.4f}")

    out_dir = args.out or Path("logs/t_predictor") / dataset_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "t_predictor.pth"
    tpred.save(model, path, extra={
        "val_mae_t": mae_t,
        "denoise_l1": denoise_l1,
        "dataset": str(dataset_path),
        "split": args.split,
        "steps": args.steps,
        "resolution": list(cfg.data.resolution),
        # The extractor is trained on the same scans that form the evaluation
        # reference, so FD/KID built on it are optimistically biased. Accepted
        # deliberately at this dataset size; recorded so it cannot be forgotten.
        "reference_overlap": True,
        **probe,
    })
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())

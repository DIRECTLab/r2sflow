"""Dataset-agnostic feature extraction and distribution comparison.

`evaluate.py` does this already, but hardcoded: its reference set is always
`kitti_360` via `DefaultConfig()`, and the depth scales are LiDARGen constants at
module level. Both assumptions are wrong for a 0.1-30 m indoor sensor -- its
`MIN_DEPTH = 0.5` alone discards 68% of our valid pixels. This module is the same
extraction path with those constants lifted into `MetricScales` and the reference
taken as an argument.

Everything here is reused by all evaluation arms; the arms differ only in how the
(depth, xyz, reflectance, mask) stream is produced.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from . import bev, distribution, extractor

# Per-metric feature/aggregation options, from evaluate.py:75-79.
METRIC_CONFIGS = {
    "FRD": dict(feature="penultimate", agg_type="subsample"),
    "FRID": dict(feature="penultimate", agg_type="depth"),
    "FPD": dict(feature=None, agg_type=None),
    "FSVD": dict(feature="penultimate", agg_type="depth"),
    "FPVD": dict(feature="penultimate", agg_type="depth"),
}


# =====================================================================================
# Scales
# =====================================================================================
@dataclass
class MetricScales:
    """Every depth-dependent constant the metric suite bakes in.

    The defaults here are the repo's KITTI values so behaviour is unchanged if a
    caller does not override; `from_cfg` is what adapts them to a checkpoint.
    """

    min_depth: float = 0.5
    max_depth: float = 63.0
    fpd_norm: float = 80.0  # evaluate.py's KITTI_MAX_DEPTH
    bev_field_size: float = 160.0
    bev_bins: int = 100
    bev_min_depth: float = 3.0
    bev_max_depth: float = 70.0
    voxel_size: float = 0.05
    rangenet_mean: list[float] | None = None  # None = SemanticKITTI defaults
    rangenet_std: list[float] | None = None

    @classmethod
    def from_cfg(cls, cfg) -> "MetricScales":
        """Derive scales from a checkpoint's data config."""
        lo, hi = float(cfg.data.min_depth), float(cfg.data.max_depth)
        return cls(
            min_depth=lo,
            max_depth=hi,
            fpd_norm=hi,
            # BEV spans a square of +-max_depth, and the depth gate must match the
            # data or the histogram comes out essentially empty.
            bev_field_size=2.0 * hi,
            bev_min_depth=lo,
            bev_max_depth=hi,
        )

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


# =====================================================================================
# Scan sources
# =====================================================================================
class ConverterScans(torch.utils.data.Dataset):
    """Scans straight from a converter's output directory.

    Generalises `load_scan_batch` in tools/visualize_flow.py. Used for datasets
    that have no HuggingFace builder (e.g. go2_real_l1).
    """

    def __init__(self, root: Path, resolution, split: str = "all", select=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import lidar_grid

        self.lidar_grid = lidar_grid
        self.root = Path(root)
        self.info = json.loads((self.root / "sensor_info.json").read_text())
        height, width = resolution
        self.grid = lidar_grid.RayGrid(
            height, width, self.info["h_up_deg"], self.info["h_down_deg"]
        )
        frames = json.loads((self.root / "frames.json").read_text())
        if split != "all":
            frames = [f for f in frames if f.get("split") == split]
        if select is not None:
            frames = [frames[i] for i in select]
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        path = self.root / "velodyne_points" / "data" / frame["file_name"]
        xyz, reflectance = self.lidar_grid.load_scan(path)
        image = self.grid.project(
            xyz, reflectance, self.info["min_depth"], self.info["max_depth"]
        )
        g = self.lidar_grid
        take = lambda c: torch.from_numpy(image[..., c]).float()[None]
        return {
            "depth": take(g.CH_DEPTH),
            "xyz": torch.from_numpy(image[..., [g.CH_X, g.CH_Y, g.CH_Z]]).float().permute(2, 0, 1),
            "reflectance": take(g.CH_REFLECTANCE),
            "mask": take(g.CH_MASK),
        }


def open_scans(spec, resolution, split="all", select=None, num_workers=4):
    """Open a scan source, dispatching on what the path actually contains.

    - a HuggingFace builder directory (`<spec>/<name>.py`) -> `load_dataset`. Use
      this for the reference: it is byte-identically what train.py fed the model.
    - a converter output directory (`<spec>/sensor_info.json`) -> `ConverterScans`.
    """
    spec = Path(spec)
    builder = spec / f"{spec.name}.py"
    if builder.exists():
        import datasets as ds

        split_map = {"train": ds.Split.TRAIN, "test": ds.Split.TEST, "all": ds.Split.ALL}
        height, width = resolution
        dataset = ds.load_dataset(
            path=str(spec),
            name=f"spherical-{height}x{width}",
            split=split_map[split],
            num_proc=num_workers,
            trust_remote_code=True,
        ).with_format("torch")
        if select is not None:
            dataset = dataset.select(select)
        return dataset
    if (spec / "sensor_info.json").exists():
        return ConverterScans(spec, resolution, split=split, select=select)
    raise SystemExit(
        f"{spec} is neither a HuggingFace builder directory (expected "
        f"{builder.name}) nor a converter output (expected sensor_info.json)"
    )


# =====================================================================================
# Extraction
# =====================================================================================
def _resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")


def build_extractor(resolution, scales, metrics, compile=False):
    proj = extractor.FeatureExtractor(
        resolution, metrics=tuple(metrics), compile=compile, scales=scales
    )
    proj.eval()
    return proj


@torch.no_grad()
def extract_batch(proj, batch, scales, device, metrics, tpred=None, model_input=None):
    """One batch of (depth, xyz, reflectance, mask) -> per-metric features.

    This is evaluate.py:85-120 with the LiDARGen constants replaced by `scales`.
    """
    height, width = proj.resolution if hasattr(proj, "resolution") else batch["depth"].shape[-2:]
    depth = _resize(batch["depth"], (height, width)).to(device)
    xyz = _resize(batch["xyz"], (height, width)).to(device)
    reflectance = _resize(batch["reflectance"], (height, width)).to(device)
    mask = _resize(batch["mask"], (height, width)).to(device)
    mask = mask * torch.logical_and(depth > scales.min_depth, depth < scales.max_depth)

    img = torch.cat([depth, xyz, reflectance], dim=1) * mask
    pcd = einops.rearrange(img[:, 1:4], "B C H W -> B (H W) C")

    out = {}
    if "FRD" in metrics:
        out["FRD"] = proj(img, metrics="FRD", **METRIC_CONFIGS["FRD"]).cpu()
    if "FRID" in metrics:
        out["FRID"] = proj(img[:, :4], metrics="FRID", **METRIC_CONFIGS["FRID"]).cpu()
    if "FPD" in metrics:
        scaled = pcd.transpose(1, 2) / scales.fpd_norm
        out["FPD"] = proj(scaled, metrics="FPD", **METRIC_CONFIGS["FPD"]).cpu()
    if "FSVD" in metrics:
        out["FSVD"] = proj(pcd, metrics="FSVD", **METRIC_CONFIGS["FSVD"]).cpu()
    if "FPVD" in metrics:
        out["FPVD"] = proj(pcd, metrics="FPVD", **METRIC_CONFIGS["FPVD"]).cpu()
    if tpred is not None:
        if model_input is None:
            raise ValueError("the timestep predictor needs the normalised model tensor")
        out["FTD"] = tpred.features(model_input.to(device)).cpu()

    hists = torch.stack(
        [
            bev.point_cloud_to_histogram(
                point_cloud,
                field_size=scales.bev_field_size,
                bins=scales.bev_bins,
                min_depth=scales.bev_min_depth,
                max_depth=scales.bev_max_depth,
            )
            for point_cloud in pcd
        ]
    ).cpu()
    return out, hists, mask.mean().item()


@dataclass
class FeatureSet:
    """Extracted features plus the scan index each row came from.

    `scan_id` is what makes cluster/block bootstrapping possible: several rows can
    share a scan (K noise draws), and those rows are not independent samples.
    """

    feats: dict[str, np.ndarray] = field(default_factory=dict)
    hists: np.ndarray | None = None
    scan_id: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def __len__(self):
        return 0 if self.scan_id is None else len(self.scan_id)

    def num_scans(self):
        return 0 if self.scan_id is None else len(np.unique(self.scan_id))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path: Path) -> "FeatureSet":
        return pickle.load(open(path, "rb"))


def accumulate(chunks) -> FeatureSet:
    """Collate per-batch outputs into one FeatureSet."""
    feats = defaultdict(list)
    hists, scan_ids = [], []
    for per_metric, hist, ids in chunks:
        for key, value in per_metric.items():
            feats[key].append(value)
        hists.append(hist)
        scan_ids.append(np.asarray(ids))
    return FeatureSet(
        feats={k: torch.cat(v, dim=0).numpy() for k, v in feats.items()},
        hists=torch.cat(hists, dim=0).numpy(),
        scan_id=np.concatenate(scan_ids),
    )


# =====================================================================================
# Reference-fitted projection
# =====================================================================================
class ReferenceProjection:
    """Standardise, PCA, then whiten -- all fitted on the REFERENCE set only.

    Fitting on pooled data would let each arm influence the space it is measured
    in. Reducing dimension is not cosmetic: Frechet distance between two samples
    of the *same* distribution grows roughly as d^2/(2n), so at n=32 a 2048-d
    feature has a null floor in the thousands and measures dimension, not
    distribution.

    The whitening step is load-bearing, not tidiness. `compute_squared_mmd` uses a
    degree-3 polynomial kernel, `(x . y / d + 1)^3`. PCA coordinates carry the full
    eigenvalue variance, which for correlated features runs to the hundreds, and
    cubing that overflows the metric into the millions -- a same-distribution NULL
    came out at 2e6 before this was added. Scaling each component to unit variance
    on the reference keeps the kernel argument O(1) and makes the distance
    isotropic in the reference's own metric, so no single loud direction dominates.
    """

    def __init__(self, reference: np.ndarray, dim: int, shrinkage: float = 0.01):
        self.mean = reference.mean(axis=0)
        self.std = reference.std(axis=0) + 1e-8
        z = (reference - self.mean) / self.std
        centred = z - z.mean(axis=0)
        self.dim = int(min(dim, centred.shape[1], max(1, centred.shape[0] - 1)))
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        self.components = vt[: self.dim]
        variance = singular[: self.dim] ** 2 / max(1, centred.shape[0] - 1)
        # Shrink toward the leading eigenvalue rather than whitening each component
        # exactly. This reference has a participation ratio of only ~4-9 (true of
        # the pretrained extractors too -- it is one trajectory in one room), so
        # the trailing components carry almost no variance and exact whitening
        # amplifies them by orders of magnitude; corrupted inputs then projected
        # onto them gave ladder values around 1e21. Shrinkage caps the
        # amplification at 1/sqrt(shrinkage) relative to the top component.
        self.variance = variance
        self.whiten = 1.0 / np.sqrt(variance + shrinkage * float(variance[0]))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean) / self.std
        return (z @ self.components.T) * self.whiten


def participation_ratio(feats: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the feature covariance.

    A health check on a learned extractor: very low means the representation
    collapsed onto one direction, very high with a flat spectrum means it is
    emitting near-orthogonal per-sample codes, i.e. memorising.
    """
    eigenvalues = np.linalg.eigvalsh(np.cov(feats, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    return float(total**2 / np.square(eigenvalues).sum()) if total > 0 else 0.0


def intraclass_correlation(feats: np.ndarray, scan_id: np.ndarray) -> float:
    """Fraction of feature variance that is between-scan rather than within-scan.

    Drives the design effect: K noise draws of one scan are not K independent
    samples, and N_eff = N*K / (1 + (K-1)*ICC).
    """
    groups = [feats[scan_id == s] for s in np.unique(scan_id)]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        return float("nan")
    within = np.mean([g.var(axis=0, ddof=1).mean() for g in groups])
    between = np.var([g.mean(axis=0) for g in groups], axis=0, ddof=1).mean()
    total = within + between
    return float(between / total) if total > 0 else float("nan")


# =====================================================================================
# Comparison
# =====================================================================================
def _block_subsample(scan_ids: np.ndarray, block: int, rng, fraction: float = 0.7) -> np.ndarray:
    """Draw a fraction of contiguous scan blocks, WITHOUT replacement.

    Deliberately subsampling rather than the usual with-replacement bootstrap.
    KID is a U-statistic over pairs, and a duplicated scan contributes an
    off-diagonal pair with itself -- a maximal kernel value -- which biases the
    estimate upward. With replacement the resulting interval did not even contain
    its own point estimate (KID -0.014, CI [0.10, 0.45]). Subsampling keeps every
    drawn scan distinct; the interval is for a slightly smaller n and so is
    conservative, which is the right direction to err here.

    Blocks rather than individual scans because the scans come from one continuous
    trajectory and neighbours are strongly correlated.
    """
    unique = np.unique(scan_ids)
    blocks = [unique[i : i + block] for i in range(0, len(unique), block)]
    count = max(1, int(round(fraction * len(blocks))))
    chosen = rng.choice(len(blocks), size=count, replace=False)
    return np.concatenate([blocks[i] for i in sorted(chosen)])


def _rows_for(scan_ids: np.ndarray, picked: np.ndarray) -> np.ndarray:
    """Row indices for a resampled set of scans, keeping each scan's draws together."""
    index = {s: np.flatnonzero(scan_ids == s) for s in np.unique(scan_ids)}
    return np.concatenate([index[s] for s in picked if s in index])


def fit_projections(reference: FeatureSet, metrics, kid_dim: int, fd_dim: int) -> dict:
    """Fit the KID/FD projections once, on the FULL reference.

    Must be done once and shared, not refitted per comparison. Fitting on a subset
    (e.g. one half of the reference for the NULL) whitens directions whose small
    eigenvalues are estimated from that subset alone; anything else then projects
    onto them with large magnitude and the cubic kernel explodes -- a NULL that
    should have been ~0 measured 2.5e4 this way. Sharing one space is also what
    makes the arms comparable to each other at all.
    """
    return {
        metric: (
            ReferenceProjection(reference.feats[metric], kid_dim),
            ReferenceProjection(reference.feats[metric], fd_dim),
        )
        for metric in metrics
        if metric in reference.feats
    }


def compare(
    reference: FeatureSet,
    arm: FeatureSet,
    metrics,
    projections: dict,
    bootstrap: int = 0,
    block: int = 8,
    num_subsets: int = 500,
    seed: int = 0,
) -> dict:
    """Score one arm against the reference, in a pre-fitted feature space.

    KID is reported first and is the primary number: it estimates no covariance
    and so carries no sample-size bias, whereas FD's bias at our n is larger than
    any effect we could measure. FD is kept for continuity with the paper, at low
    dimension and always beside its own measured null.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for metric in metrics:
        if metric not in reference.feats or metric not in arm.feats:
            continue
        if metric not in projections:
            continue
        ref_raw, arm_raw = reference.feats[metric], arm.feats[metric]

        kid_proj, fd_proj = projections[metric]
        ref_kid, arm_kid = kid_proj(ref_raw), kid_proj(arm_raw)
        ref_fd, arm_fd = fd_proj(ref_raw), fd_proj(arm_raw)

        entry = {
            "KID": float(distribution.compute_squared_mmd(ref_kid, arm_kid, num_subsets=num_subsets)),
            "FD": float(distribution.compute_frechet_distance(ref_fd, arm_fd)),
            "kid_dim": kid_proj.dim,
            "fd_dim": fd_proj.dim,
            "n_ref_rows": len(ref_raw),
            "n_arm_rows": len(arm_raw),
            "n_ref_scans": reference.num_scans(),
            "n_arm_scans": arm.num_scans(),
            "fd_rank_deficient": fd_proj.dim >= min(len(ref_raw), len(arm_raw)),
        }
        if arm.scan_id is not None:
            entry["icc"] = intraclass_correlation(arm_kid, arm.scan_id)
            k = len(arm_raw) / max(arm.num_scans(), 1)
            icc = entry["icc"] if np.isfinite(entry["icc"]) else 1.0
            entry["n_eff"] = float(len(arm_raw) / (1 + (k - 1) * icc))

        if bootstrap:
            kid_samples, fd_samples = [], []
            for _ in range(bootstrap):
                r = _rows_for(reference.scan_id, _block_subsample(reference.scan_id, block, rng))
                a = _rows_for(arm.scan_id, _block_subsample(arm.scan_id, block, rng))
                kid_samples.append(
                    distribution.compute_squared_mmd(ref_kid[r], arm_kid[a], num_subsets=64)
                )
                fd_samples.append(distribution.compute_frechet_distance(ref_fd[r], arm_fd[a]))
            entry["KID_ci"] = [float(np.percentile(kid_samples, 2.5)), float(np.percentile(kid_samples, 97.5))]
            entry["FD_ci"] = [float(np.percentile(fd_samples, 2.5)), float(np.percentile(fd_samples, 97.5))]
            entry["ci_method"] = "block subsampling without replacement (70% of blocks)"
        results[metric] = entry

    if reference.hists is not None and arm.hists is not None:
        results["BEV"] = {
            "JSD": float(bev.compute_jsd_2d(torch.from_numpy(reference.hists), torch.from_numpy(arm.hists))),
            "MMD": float(bev.compute_mmd_2d(torch.from_numpy(reference.hists), torch.from_numpy(arm.hists))),
        }
    return results


def subset_scans(features: FeatureSet, picked: np.ndarray) -> FeatureSet:
    rows = _rows_for(features.scan_id, picked)
    return FeatureSet(
        feats={k: v[rows] for k, v in features.feats.items()},
        hists=None if features.hists is None else features.hists[rows],
        scan_id=features.scan_id[rows],
    )


def split_halves(
    features: FeatureSet, block: int = 8, mode: str = "interleaved", seed: int = 0
) -> tuple[FeatureSet, FeatureSet]:
    """Split the reference against itself, for a NULL floor.

    Three modes, because on a single continuous trajectory they measure different
    things and the difference is itself informative:

    - `random`    ignores time entirely -- pure sampling noise, so this is the
                  plumbing check. Should sit at ~0.
    - `interleaved` alternating blocks -- mild temporal structure.
    - `contiguous`  first half vs second half -- includes slow scene drift.

    Measured on go2w_sim_l1 these came out at roughly 0.02 / 0.07 / 0.18 (KID,
    FSVD), so a reference-vs-reference comparison is *not* zero and the arms have
    to be read against whichever floor matches their own temporal relationship to
    the reference. An arm seeded from a disjoint later block is a contiguous
    relationship, not an interleaved one.
    """
    unique = np.unique(features.scan_id)
    if mode == "random":
        picked = np.random.default_rng(seed).permutation(unique)
        half = len(picked) // 2
        first, second = picked[:half], picked[half:]
    elif mode == "contiguous":
        half = len(unique) // 2
        first, second = unique[:half], unique[half:]
    elif mode == "interleaved":
        blocks = [unique[i : i + block] for i in range(0, len(unique), block)]
        first = np.concatenate(blocks[0::2]) if blocks[0::2] else np.array([])
        second = np.concatenate(blocks[1::2]) if blocks[1::2] else np.array([])
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return subset_scans(features, first), subset_scans(features, second)

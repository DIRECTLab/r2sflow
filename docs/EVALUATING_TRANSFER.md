<h1>Evaluating real→sim transfer</h1>

A quantitative pipeline for judging whether a change to the model, the ray grid, or
the way we generate sim data actually helped — instead of eyeballing
[VISUALIZING_FLOW.md](VISUALIZING_FLOW.md).

- [What it measures](#what-it-measures)
- [Relationship to the paper](#relationship-to-the-paper)
- [The NULL floors](#the-null-floors)
- [Running it](#running-it)
- [Caveats](#caveats)

## What it measures

Every arm is scored against the same sim REFERENCE distribution:

|arm|how the scans are produced|
|:-|:-|
|**REFERENCE**|sim scans, straight through, no model|
|**NULL**|the reference split against itself — the floor|
|**A** "basic generation"|sim seed → noise to `t0` → integrate to 1|
|**A′**|as A, with fill randomly matched to real|
|**B** "real→sim"|real seed → the same treatment|

Arms differ *only* in the source path; the transport, decode and extraction code is
shared. Noise is seeded by draw index alone, so A and B see identical noise tensors
at the same `(scan, draw)` — the comparison is paired rather than two independent
Monte Carlo runs. At `t0=0` the seed is fully discarded, so A and B become the same
distribution by construction; that is a falsifiable check for seed leaks.

**A′ is not optional.** Sim scans are ~80% filled and real ones ~39%, with zero
overlap and 27× fewer points, and an empty pixel is exactly −1.0 after log-scale
encoding. Any A-vs-B gap is therefore explainable by point density alone. A′ is what
separates "the domains differ" from "one has more points".

## Relationship to the paper

`~/research/lidar_s2r/lidar_flow.pdf` §IV.A:

| |paper|here|
|:-|:-|:-|
|feature extractors|**none trained** — off-the-shelf RangeNet/PointNet/MinkowskiNet/SPVCNN|train one (`FTD`), plus the retuned pretrained ones|
|extractor domain|real LiDAR, evaluating a real-LiDAR generator|sim, evaluating a sim generator — same relationship|
|sampler|`euler`, fixed step size|same|
|samples|10,000 generated vs 81,106 reference|32 vs 118|
|range encoding|`log(x+1)/log(x_max+1)`|same|

The paper's extractors are SemanticKITTI outdoor models. Our data is indoor with a
0.34 m median depth, so every depth-dependent constant in the suite is wrong for it —
`evaluate.py`'s `MIN_DEPTH = 0.5` alone discards **68%** of valid pixels, and
`bev.py`'s `min_depth = 3.0` discards **~98%** of points. `MetricScales.from_cfg`
derives them from the checkpoint instead.

## The NULL floors

A reference-vs-reference score is **not zero**, because the sim scans are one
continuous trajectory rather than independent draws. Measured on `go2w_sim_l1`
(n=118, KID):

|split|FRID|FPD|FSVD|FPVD|
|:-|:-|:-|:-|:-|
|**random** scan split|−0.016|0.012|0.020|0.023|
|**interleaved** blocks|−0.014|0.061|0.071|0.064|
|**contiguous** halves|0.047|0.296|0.175|0.133|

The ordering random < interleaved < contiguous is the signature of temporal
non-stationarity, and all three are reported:

- **random** ignores time, so it is pure sampling noise — the plumbing check. It
  should sit at ~0, and does.
- **contiguous** is the floor an arm should usually be read against: Arm A's seeds
  are a disjoint *later* block of the same trajectory, which is a contiguous
  relationship, not an interleaved one.

Reading an arm against zero rather than against the matching floor would credit the
model with a gap that is really just scene drift.

## Running it

Everything runs inside the container.

**1. Train the feature extractor.** The pretext task is recovering the flow timestep
`t` from `x_t`, using the same `torchcfm` flow matcher train.py uses, built from the
checkpoint's own config.

```bash
python tools/train_t_predictor.py \
    --ckpt logs/r2flow-1rf/go2w_sim_l1/spherical-64x512/.../checkpoint_0002560000.pth \
    --dataset r2flow/data/go2w_sim_l1 --split all \
    --probe-dataset r2flow/data/go2_real_l1/dataset \
    --steps 20000 --out logs/t_predictor/go2w_sim_l1
```

Two design points that are load-bearing rather than decorative:

- **The denoising head.** t-prediction alone is close to trivial — the local pixel
  variance of `x_t` is about `(1-t)²`, so a shallow network solves it with a single
  feature direction, and a rank-1 representation is useless as a distribution
  metric. The denoise head (largest loss weight) forces the encoder to keep scene
  content; the 32-bin classification head forces ≥31 independent directions before
  the head can be accurate.
- **Valid-pixel dropout, `p ~ U(0, 0.6)`.** Without it the encoder learns to read
  fill fraction, and since sim and real fill do not overlap at all, every arm
  comparison would collapse to counting points.

**2. Check the floors** (no arms, no extractor needed):

```bash
python evaluate_flow.py --ckpt <checkpoint> \
    --reference r2flow/data/go2w_sim_l1 --reference-split train --reference-frames 0:118 \
    --metrics FRID FPD FSVD FPVD --null-only
```

**3. Full run:**

```bash
python evaluate_flow.py --ckpt <checkpoint> \
    --reference r2flow/data/go2w_sim_l1 --reference-split train --reference-frames 0:118 \
    --arm-a r2flow/data/go2w_sim_l1 --arm-a-split test \
    --arm-b r2flow/data/go2_real_l1/dataset --arm-b-split test \
    --t-predictor logs/t_predictor/go2w_sim_l1/t_predictor.pth \
    --t 0 0.3 0.6 0.8 1.0 --repeats 8 \
    --export logs/eval/transfer.json
```

The reference stops at frame 117 and Arm A's seeds start at 126, leaving an 8-frame
guard band — about one decorrelation length. Without it Arm A at high `t0` would be
a near-copy of a reference item and score at the floor by construction.

## What the first run found

Recorded because these shape how the numbers should be read, not as results in
their own right (158 sim scans, N_eff ≈ 40 — nothing here decides anything).

**The metrics are not equally trustworthy on this data.** From the corruption
ladders (`--ladder`):

|metric|noise ladder|dropout ladder|
|:-|:-|:-|
|FRID|ρ=+1.00|ρ=+1.00|
|FSVD|ρ=+1.00|ρ=+1.00|
|FPVD|ρ=+1.00|ρ=+1.00|
|**FPD**|ρ=+1.00|**ρ=−0.90 — inverted**|
|**FTD**|**ρ=+0.90 — saturates**|ρ=+1.00|

FPD gets *closer* to the reference as points are dropped (0.132 → 0.068). Since the
sim/real gap is largely density, FPD reads backwards on Arm B and should be
discounted. FTD is monotone from σ=0.01 to 0.10 then saturates at 0.20, which is
expected of an extractor trained to read noise level.

**KID magnitudes are only interpretable near the reference.** The kernel is cubic
and unbounded, so inputs far outside the reference produce values in the thousands
or worse (Arm B's FTD at `t0=1` is 1.4e8). Rankings near the floor are meaningful;
absolute values far from it are not.

**The 1-RF model degrades data when integrated near t=1.** Arm A's score is *worse*
at high `t0` than leaving the scan untouched:

|t0|Euler steps|FRID|
|:-|:-|:-|
|0.995|1|0.098|
|0.99|3|0.157|
|0.95|13|5.27|
|0.9|26|3.52|
|1.0|0 (identity)|0.107|

At one step the transport is exact — it lands on the contiguous NULL (0.042), which
is what rules out a bug in the transport or decode path. Degradation scales with the
number of steps taken near t=1. This matches the paper's own Fig. 8: 1-RF has high
trajectory curvature at both ends, and "the raydrop pixels drifting toward a value of
−1 may hinder the training of straight flows". Reflowing to 2-RF is the paper's fix.

**Participation ratio is low for every extractor**, not just the trained one: FPD
1.4, FTD 3.7, FRID 6.7, FPVD 7.8, FSVD 9.4. The pretrained SemanticKITTI models
cannot have collapsed onto a flow timestep, so this is a property of a reference that
is one trajectory in one room. Judge FTD relative to the pretrained extractors, never
against an absolute band.

## Caveats

- **KID is primary, not FD.** FD estimates a covariance; between two samples of the
  *same* distribution it grows roughly as `d²/(2n)`, which at n=32 and d=2048 is a
  floor in the thousands. It is reported at d=16 beside its own measured null. KID
  (`compute_squared_mmd`) estimates no covariance and carries no such bias.
- **Features are whitened before either metric.** The MMD kernel is cubic,
  `(x·y/d + 1)³`; unwhitened PCA coordinates carry the full eigenvalue variance and
  cubing overflows the metric — a same-distribution NULL measured 2.5e4 before this
  was added. The projection is fitted **once on the full reference** and shared by
  every arm; refitting per-comparison reintroduces the same explosion.
- **Confidence intervals are block *subsampling*, not bootstrap.** KID is a
  U-statistic over pairs, so a resampled duplicate contributes a maximal-kernel
  self-pair and biases it upward — with replacement the interval did not contain its
  own point estimate. Subsampling keeps drawn scans distinct and is conservative.
- **Never quote `n = scans × repeats`.** Extra noise draws per scan are correlated;
  the design effect is `1 + (K-1)·ICC`. The reported `n_eff` accounts for it, and the
  reference cannot be augmented at all, so it remains the binding constraint.
- **Absolute values are not comparable to the paper's table** (10,000 vs 81,106
  samples there; 32 vs 118 here) or to any published LiDAR number.
- **The extractor is currently trained on the reference set**, so FTD is
  optimistically biased. Deliberate at this dataset size and recorded in the
  checkpoint as `reference_overlap`; the first thing to fix once more sim data
  exists is to give the extractor its own partition.
- **Arm B describes one 219 s bag**, one environment, one trajectory.
- FRD is dropped whenever the checkpoint has `train_reflectance=False`, since it is
  the only metric consuming reflectance. It is dropped for the reference too — a
  score computed on one side only has no counterpart.

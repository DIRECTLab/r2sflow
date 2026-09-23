<h1>Unitree Go2-W simulator rosbags</h1>

How to turn a Go2-W simulator rosbag into R2Flow range images.

- [What the repo expects](#what-the-repo-expects)
- [What the bags contain](#what-the-bags-contain)
- [The sensor's ray grid](#the-sensors-ray-grid)
- [Building the dataset](#building-the-dataset)
- [Training](#training)
- [Caveats](#caveats)

## What the repo expects

`r2flow/data/*/` holds a HuggingFace `datasets` builder that yields, per scan:

|Feature|Shape|Contents|
|:-|:-|:-|
|`xyz`|`(3, H, W)`|cartesian coordinates, metres|
|`reflectance`|`(1, H, W)`|intensity in `[0, 1]`|
|`depth`|`(1, H, W)`|range in `[min_depth, max_depth]`, metres|
|`mask`|`(1, H, W)`|1 where a ray returned|

`train.py` builds the `C x H x W` model input from these: `depth` goes through
`LiDARUtility.convert_metric_depth` and `reflectance` through `normalize`, giving
2 channels (or 1 with `--train_reflectance False`). So the range image is the
*model input*, not the on-disk format — the dataset stores metric geometry and
the normalisation happens at train time.

Rows are elevation (row 0 topmost), columns are azimuth (column 0 at +180°,
advancing clockwise). This is the convention `get_hdl64e_linear_ray_angles`
produces, and `LiDARUtility` relies on it to convert between depth and xyz.

## What the bags contain

A recording has `*_sensors_*.bag` (split into ~512 MB parts) and one
`*_tf_*.bag`. Three topics carry point clouds; only one is usable:

|Topic|Verdict|
|:-|:-|
|`/<robot>/livox/lidar`|**Use this.** 40 rings, sensor frame, exact ray grid.|
|`/<robot>/lidar`|Unusable. 563 "rings" spanning a full 360° *vertical* circle, so it mostly scans the robot's own body — 73% of returns are closer than 1.45 m and the median range is 0.30 m.|
|`/<robot>/merged_cloud`|Unusable. Accumulated in the `odom` frame and z-clipped, so there is no single sensor origin to project from.|

The catch with the livox topic is that **one message is not one scan**. Each
message covers a ~60° azimuth sector (83 columns), and the sector order is
irregular — sectors repeat and get skipped. Covering the full 360° takes about
10 messages spanning ~0.56 s, during which the robot moves (mean 1.27 m/s, peak
2.64 m/s). `tools/rosbag_to_r2flow.py` therefore accumulates messages until the
azimuth is covered and de-skews each sector into the sensor pose at the window's
reference time, using `odom -> base` from the tf bag plus the static
`base -> livox_frame`.

De-skewing is worth it. Where two messages from different times hit the same
grid cell, their depths disagree by 0.67 m on average raw, and 0.17 m after
de-skewing — a 75% reduction (85% at the median).

## The sensor's ray grid

Measured from the bags and exact to ~1e-5°:

```
elevation(ring) = 29.50 - 1.48 * ring      ring in [0, 39]      -> 40 rows
azimuth lands on an exact 0.72 deg grid                         -> 500 columns
```

That elevation table is reproduced exactly by the repo's own linear form with
`h_up = 29.50`, `h_down = -29.70`, `H = 40`, which is what
`r2flow.utils.lidar.get_go2w_livox_linear_ray_angles` returns. `train.py` picks
it up through `SPHERICAL_RAY_ANGLES[cfg.data.dataset]`, so no angle table file is
needed.

**Use width 512, not the native 500.** Every architecture here downsamples by 8
— HDiT tokenizes by `patch_size=(1, 4)` and halves three times (needing
`H % 8 == 0` and `W % 32 == 0`), and both U-Nets hit a skip-connection size
mismatch at width 500. The `spherical-40x500` config exists for analysis only.
The cost of 512 is negligible: columns become 0.703° instead of 0.72°, and
because de-skewing jitters points off the exact grid anyway, no column ends up
systematically empty (overall fill 0.880 at 512 vs 0.888 at 500).

## Building the dataset

The tool needs only `numpy` and `rosbags` — deliberately not torch, so it runs
outside the R2Flow environment:

```bash
uv venv .venv-rosbag
VIRTUAL_ENV=$PWD/.venv-rosbag uv pip install rosbags numpy
```

```bash
.venv-rosbag/bin/python tools/rosbag_to_r2flow.py \
    --bag-dir /path/to/rosbag_YYYYMMDD_HHMMSS_NNNNNN \
    --out /path/with/space/go2w_sim_r2flow

ln -sfn /path/with/space/go2w_sim_r2flow r2flow/data/go2w_sim/dataset
```

The symlink is enough when running on the host. Inside Docker the repo is
bind-mounted, so that symlink would point at a host path the container cannot
see; set `GO2W_SIM_ROOT` instead and `compose.yml` mounts the scans over
`r2flow/data/go2w_sim/dataset` the same way it does for KITTI-360:

```bash
export GO2W_SIM_ROOT=/path/with/space/go2w_sim_r2flow
docker compose up --detach && docker compose exec r2flow bash
```

Pass `--topic`, `--odom-frame`, `--base-frame` and `--sensor-frame` if the robot
namespace differs from the default `go2w_sim_005`. `--no-compensate` skips
de-skewing entirely and needs no tf bag.

The output mirrors KITTI's layout, so the point files stay useful for anything
else that reads velodyne `.bin`:

```sh
go2w_sim_r2flow/
├── velodyne_points/data/
│   ├── 0000000000.bin      # float32 [x, y, z, reflectance] * N
│   └── ...
├── frames.json             # per-scan timing, coverage and train/test split
└── sensor_info.json        # ray grid, depth bounds, intensity scale
```

Scans are ~100 MB per 128 s of recording. The split is a contiguous temporal cut
(last 20% is test) so train and test never share a stretch of trajectory.

On a 128 s recording this yields 202 scans from 2036 messages (162 train / 40
test) at 0.88 mean pixel fill.

## Training

```bash
accelerate launch train.py \
    --dataset go2w_sim --projection spherical-40x512 --resolution 40 512 \
    --min_depth 0.5 --max_depth 80.0 \
    --batch_size 8 --loss_fn l2 --timestep_distribution uniform \
    --output_dir logs/r2flow-go2w-1rf
```

`--min_depth 0.5` matters: the default 1.45 is tuned for the HDL-64E and would
mask out 11% of this sensor's returns. Keep it equal to the tool's `--min-depth`
so the stored `mask` and `LiDARUtility.get_mask` agree. Everything else in
[TRAINING.md](TRAINING.md) — straightening, distillation, sampling — applies
unchanged.

## Caveats

- **Reflectance is a guess at scale.** Raw sim intensity runs 0.02–85.4, and the
  tool divides by `--intensity-scale` (default 100) to reach `[0, 1]`. The sim's
  units are not calibrated against KITTI's, so treat the reflectance channel as
  arbitrary units unless you match it to your target domain. Train with
  `--train_reflectance False` for range only.
- **Images are sensor-aligned, not gravity-aligned.** The lidar is statically
  pitched 13° down from `base`, so the horizon sits around row 11 of 40 and 28
  rows look downward — a more ground-heavy framing than KITTI-360, where the
  horizon is at row 7 of 64. This barely varies while walking (body pitch σ 1.1°,
  roll σ 0.8°, both under one 1.48° ring), so the sensor frame is within a row of
  gravity-aligned. To change it, de-skew to `odom -> base` instead of
  `odom -> livox_frame` in `assemble()` and widen `H_UP`/`H_DOWN` to cover the
  rotated band.
- **Scans are ~0.56 s apart and overlapping in content, not in points.** Windows
  are consecutive and non-overlapping, so 128 s of recording gives only ~200
  scans — small for generative training. Record more, or accumulate with a
  sliding window if you need more samples.
- **The mask is not a free-space label.** Empty pixels mix "no return" with "the
  robot's own body occluded the ray" (visible as a solid block at azimuth ±180°)
  and "de-skewing moved the point elsewhere".
- **`--coverage` trades completeness against smear.** Lowering it emits scans
  sooner with shorter windows and less de-skew residual, but with more holes.
  Windows that stall past `--max-window` are dropped unless they already reach
  `--min-coverage`; the tool reports how many.

<h1>Visualising the flow on real LiDAR</h1>

How to feed a real Unitree Go2 scan to a trained checkpoint and watch the flow
transport it, step by step.

- [What this actually shows](#what-this-actually-shows)
- [Module layout](#module-layout)
- [Converting a real ROS 2 bag](#converting-a-real-ros-2-bag)
- [Running the visualiser](#running-the-visualiser)
- [Reading the output](#reading-the-output)
- [Caveats](#caveats)

## What this actually shows

R2Flow is an **unconditional** generative model. `sample.py` draws
`x0 ~ N(0, I)` and integrates `dx/dt = model(t, x)` up to `t=1`; there is no
encoder and no input/output pair. So "running a scan through the model" is not
reconstruction — it is **transport**. We substitute a real scan for the usual
Gaussian sample and integrate anyway, keeping every intermediate state:

```
x_{k+1} = x_k + (1/N) * model(t_k, x_k),   t_k = k/N,   k = 0 .. N-1
```

With `N=8` that gives 9 states: `t=0` is the untouched input, `t=8` the final
output. The further along, the more the scan has been reshaped toward the
distribution the checkpoint was trained on. How *little* it moves is a direct
read on whether the input is in-distribution.

That contrast is the useful part. Run it on the sim data the model was trained
on and the scene survives all 8 steps — walls, ground-plane gradient and
doorways stay legible, just progressively smoothed. Run it on the real Go2 bag
and the structure is gone by `t=6`, replaced with a smooth sim-like gradient.

## Module layout

Split so the bag parsing never drags torch in, and the projection is defined
exactly once:

|Module|Needs|Role|
|:-|:-|:-|
|`tools/lidar_grid.py`|numpy|`RayGrid` + spherical projection + scan I/O. Pure geometry, importable from either environment.|
|`tools/bag_io.py`|rosbags|PointCloud2 decoding and message iteration, ROS 1 and ROS 2.|
|`tools/rosbag_to_r2flow.py`|rosbags|Sim (Livox) converter — ring-indexed, tf de-skewed.|
|`tools/rosbag2_go2_to_r2flow.py`|rosbags|Real (utlidar L1) converter — sweep accumulation, origin check.|
|`r2flow/utils/flow.py`|torch|`euler_trajectory`, plus `encode`/`split_channels` mirroring train.py.|
|`tools/visualize_flow.py`|torch, matplotlib|CLI: scans -> trajectory -> figure.|

`r2flow/data/*/**.py` keeps its own copy of the projection because the
`datasets` library loads builders from an isolated cache directory and they
cannot import from `tools/`. Those copies assert against `sensor_info.json`, so
a drift shows up as an error rather than silently wrong images.

## Converting a real ROS 2 bag

The Go2's `utlidar` L1 is a spinning non-repetitive scanner. One PointCloud2
message is a ~65 ms slice of a continuous spiral, not a sweep, and its `ring`
field is a constant, so there is no row structure to exploit — the converter
accumulates several messages and projects onto the model's ray grid.

```bash
.venv-rosbag/bin/python tools/rosbag2_go2_to_r2flow.py \
    --bag /mnt/fast/lidar_data/go2_bags/go2_mocam_loop \
    --out r2flow/data/go2_real/dataset \
    --messages-per-scan 8
```

`--messages-per-scan` trades coverage against smear. There is no `/tf` and no
odometry for the lidar frame in this bag, so accumulated sweeps cannot be
de-skewed:

|messages/scan|window|pixel fill|
|:-|:-|:-|
|1|0.00 s|2.3%|
|4|0.19 s|8.8%|
|**8** (default)|**0.45 s**|**17.7%**|
|16|0.97 s|28.5%|

On startup the tool checks that the cloud really is sensor-centred, because a
spherical projection from the wrong origin is wrong in a way that is hard to
see by eye. It compares candidate origins on two signals — the near-field void
radius, and how smoothly elevation advances through one message's sweep:

```
origin check on a single message
  (0,0,0)        void_radius=0.256 m  elevation_step_p99=2.04 deg
  (0,0,+0.345)   void_radius=0.001 m  elevation_step_p99=29.40 deg
  (0,0,-0.345)   void_radius=0.503 m  elevation_step_p99=3.75 deg
```

`(0,0,0)` wins on both, so this bag's frame is sensor-centred and needs no
correction. If yours disagrees the tool says so; override with
`--sensor-origin X Y Z`. The check deliberately runs on a **single message**:
the per-point `time` field restarts every message, so on an accumulated sweep
the elevation trace tears no matter which origin is right.

## Running the visualiser

Inside the container:

```bash
python tools/visualize_flow.py \
    --ckpt logs/r2flow-go/go2w_sim/spherical-40x512/.../checkpoint_0002560000.pth \
    --scans r2flow/data/go2_real/dataset \
    --num-samples 4 --num-steps 8 \
    --out logs/flow_viz
```

Point `--scans` at either converter's output; the grid is read back from
`sensor_info.json` and checked against the checkpoint's resolution. Use
`--sample-ids` for specific frames.

## Reading the output

One PNG per sample: nine range images stacked top to bottom (`t=0` .. `t=8`),
then the same nine states as bird's-eye point clouds coloured by surface normal.
Both come from the same code path as `log_images` in train.py, so they are
directly comparable with TensorBoard.

The one deviation is the display scale. The model normalises depth by
`max_depth = 80 m`, which is right for training but useless for looking at an
indoor scan — an 8 m room renders in the bottom 10% of the colour map and
collapses to a handful of pixels in the BEV. The visualiser instead scales to
the input's 99th-percentile depth and prints what it picked:

```
display: depth_clip=16.0 m  bev_range=16.0 m (input depth p50=1.13 max=21.37)
```

Override with `--depth-clip` / `--bev-range`. This affects display only; the
tensor handed to the model is untouched.

## Caveats

- **The real bag is far out of distribution.** The checkpoint was trained on
  simulated Livox scans: 40 uniform rings, −28.2°..+29.5°, ranges to 70 m. The
  real L1 sweeps +0.5°..+89.8° with a median range of 0.42 m and 93% of returns
  inside 1.45 m, most of them the robot's own body. Only about 25% of real
  points land in the model's elevation band *and* range window, which is why the
  input fill is ~18% against ~90% for sim. Treat the outputs as a domain-gap
  probe, not as translation.
- **Real sweeps are smeared.** No `/tf`, so the 0.45 s accumulation window is
  uncompensated. The sim converter de-skews; this one cannot.
- **Reflectance scales are unrelated.** Sim intensity is 0–85 (divided by 100),
  real L1 is 0–255 (divided by 255). Neither is calibrated against the other.
- **Euler, not `dopri5`.** `sample.py` uses an adaptive solver; this uses fixed
  Euler steps so the intermediate states are evenly spaced and meaningful. With
  few steps the endpoint will differ from what `sample.py` produces.
- **`/go2/point_cloud2` is not a scan.** It is an accumulated local map in the
  `odom` frame capped at ~3.5 m, with no single sensor origin to project from.
  Use `/go2/raw_lidar`.

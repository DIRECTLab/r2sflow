<h1>Fine-tuning a pretrained R2Flow on simulated navigation bags</h1>

How to adapt the released KITTI-360 model to scans from simulated navigation
episodes (ROS 2 bags with `/registered_scan`).

- [The data](#the-data)
- [Building the dataset](#building-the-dataset)
- [Training](#training)
- [Caveats](#caveats)

## The data

Each episode is a directory holding `episode.json` and a ROS 2 sqlite bag in
`bag/`. The only point cloud topic is `/registered_scan`:

- one message is a **full 360° scan** (~310k points, about one every 1.2 s), so
  there is no sector stitching or de-skewing to do;
- the points are registered into **`map`**, but `/tf` carries
  `map -> sensor_at_scan` with exactly the scan's stamp, so each scan is moved
  back into the sensor frame with the inverse of that pose;
- the points are **xyz only** — no intensity, hence no reflectance channel.

In the sensor frame the scans span about −24° to +11° of elevation, with ranges
of 0.8 / 2.7 / 11 m at p1 / p50 / p99.

## Building the dataset

The converter needs `rosbag2_py`, so run it in a ROS 2 Humble environment rather
than the R2Flow one:

```bash
conda run -n ros2_humble_env python tools/ros2bag_registered_to_r2flow.py \
    --episodes-dir r2flow/data/simulated_bags \
    --out /path/with/space/go2w_nav_sim_r2flow

ln -sfn /path/with/space/go2w_nav_sim_r2flow r2flow/data/go2w_nav_sim/dataset
```

The last `--test-episodes` episodes (default 2, sorted by name) form the test
split, so train and test never share a trajectory. Output follows the same
KITTI-style layout as [GO2W_SIM.md](GO2W_SIM.md), with reflectance written as 0.
Inside Docker, put the output under `LIDAR_DATA_ROOT` so the symlink resolves.

The grid is **64 × 1024 over −24°…+12°**. 64 × 1024 is the KITTI-360 resolution,
which is what lets the pretrained HDiT load unchanged: its learnable positional
embedding is sized to the token grid. The ray angles themselves
(`get_go2w_nav_linear_ray_angles`) are the target sensor's, not KITTI's.

## Training

`--init_weights` starts a normal 1-RF run (your dataset, fresh noise) from
pretrained weights. It takes a checkpoint path or a release name. This is
different from `--init_ckpt`, which is the reflow/distillation stage and trains
on generated samples instead of `--dataset`.

```bash
accelerate launch train.py \
    --dataset go2w_nav_sim --projection spherical-64x1024 --resolution 64 1024 \
    --min_depth 0.5 --max_depth 30.0 --train_reflectance False \
    --init_weights r2flow-kitti360-1rf \
    --lr 2e-5 --num_images_training 400000 --num_images_lr_warmup 8000 \
    --output_dir logs/r2flow-go2w-nav-ft
```

The KITTI-360 model takes depth and reflectance. With `--train_reflectance False`,
the loader keeps only the depth channel of the two layers that depend on the
channel count (the tokenizer conv and the detokenizer linear). Every other
tensor loads as-is, and any other mismatch raises an error. Start from the `1rf`
checkpoint, not `2rf`: training on real data with fresh noise undoes reflow's
straightening anyway. Reflow the fine-tuned model afterwards as in
[TRAINING.md](TRAINING.md).

## Caveats

- **Depth normalisation changes.** The pretrained model used `logscale` over
  1.45–80 m, and this run uses 0.5–30 m, so the same normalised value means a
  different range. Fine-tuning has to re-learn that mapping.
- **Sparse images.** Mean pixel fill is about 0.36 (≈0.5 in the middle rows,
  thinning toward the top and bottom of the band), against ≈0.9 for KITTI-360.
  The scans themselves are sparse, so the model will learn the holes as well.
- **Small dataset.** Ten episodes give about 3–4k scans, and consecutive scans
  overlap heavily, so watch the samples for memorisation on long runs.

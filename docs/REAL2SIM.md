<h1>Real→sim translation with pseudo-real pairs</h1>

This maps noisy real Go2 lidar (raydrop, ghost/multipath points, range noise, robot-body hits) onto the
clean hospital-sim distribution. It is DRUM ([arXiv 2603.26263](https://arxiv.org/abs/2603.26263))
run in reverse, built on R2Flow rectified flows.

1. **Real prior.** An unconditional R2Flow is fine-tuned from KITTI-360 on real Go2 sweeps.
2. **Pseudo-real pairs.** Each hospital sim scan is "realified" by guided sampling through the real
   prior. Raydrop and ghosts come from the prior, and geometry comes from the sim scan.
3. **Conditional model.** An HDiT is trained to map pseudo-real → sim. It takes the condition image
   as an extra input channel and is initialised from a sim prior.
4. **Deployment.** `tools/real2sim_node.py` runs it on the utlidar stream, either as a ROS 2 node or
   as an offline bag replay.

Everything runs as one Slurm chain: `bash scripts/real2sim_pipeline.sh`. With `SMOKE=1` it runs a
minutes-long version first.

## The shared ray grid

| | elevation band | source |
|:-|:-|:-|
| hospital sim (`go2w_nav_sim`) | −24°..+12° | `/registered_scan`, full scans |
| real utlidar (L2), z-flipped | 0°..−90° | `/go2/raw_lidar`, 8-message sliding window |
| **common grid** | **−24°..0°, 64×1024** | `go2w_common_sim`, `go2_common_real` |

The common grid is the intersection of the two bands. It stays 64×1024 so the KITTI-360 HDiT weights
load unchanged. About 22% of real points beyond 0.5 m land in it, which gives ~10% pixel fill.

`go2w_common_sim` needs no conversion step. `go2w_nav_sim/dataset` stores every sim point, not one
point per pixel, so the builder re-projects those files directly. The real sweeps come from:

```bash
python tools/rosbag2_go2_to_r2flow.py --bag r2flow/data/rosbags/go2_mocam_loop \
    --out r2flow/data/go2_common_real/dataset --height 64 --width 1024 \
    --h-up 0 --h-down -24 --min-depth 0.5 --max-depth 30 --messages-per-scan 8 --stride 2
```

`--stride 2` is a sliding window. It gives 1695 sweeps from the 220 s bag; non-overlapping windows
would give only 424. The last 20% of sweeps are held out as the test split.

### L2-emulating sim (`go2w_l2_sim`)

The original sim used Isaac Sim's `Example_Rotary` preset: horizontal rings, mounted level, about
twice the real fill. The real L2 works differently, as measured from `/go2/raw_lidar`:
- a mirror sweeps **vertical** arcs from 0 to 90° elevation, about 145 points per arc;
- the arc plane rotates in azimuth at about 4.3 rev/s, at 62k pts/s;
- the sensor is mounted upside down. The floor normal in the sensor frame is tilted 13.4° toward
  azimuth 123.7°, and the floor is 0.332 m below the sensor.

Accumulated sweeps are therefore irregular vertical stripes.

`../lidargen` now has a `Unitree_L2` RTX profile (`src/lidargen/sensors/unitree_l2.py`):
- a rotary model with one tick per half-arc, 440 ticks/s and a 4 Hz rotor;
- the rotor-during-arc lean is baked into the emitter azimuths;
- 0.3° elevation jitter;
- a random rotor phase after every scan, so coverage does not repeat;
- the calibrated mount above.

It is the default `lidar_config` of `go2_explore`. That env also publishes raw sensor-frame scans
on `/lidar_points`.

The dataset is built by re-rendering the **same recorded trajectories** as `go2w_nav_sim`. Only the
lidar changes, and no autonomy stack is needed:

```bash
python tools/extract_sim_poses.py                 # /state_estimation -> poses.npz per episode
bash scripts/replay_sim_l2.sh                     # Isaac replay (Slurm array) + tools/sim_l2_to_r2flow.py
SIM_DATASET=go2w_l2_sim REAL_FROM=logs/real2sim/<run> bash scripts/real2sim_pipeline.sh
```

`tools/sim_l2_to_r2flow.py` reuses the real converter's sweep accumulation, z-flip and `RayGrid`. A
window of 2 sim scans is 0.5 s, about the real 8 messages. On one test episode, compared with
real:

| | real | L2 sim | rotary sim |
|:-|:-|:-|:-|
| fill | 0.104 | 0.113 | 0.247 |
| share of returns under 1 m | 0.30 | 0.32 | 0.07 |
| columns with any return | 0.45 | 0.39 | 0.92 |

Known gaps:
- RTX drops returns nearer than about 0.4 m whatever `nearRangeM` says. This is harmless because
  both datasets cut at 0.5 m.
- The replay has no robot body.
- The real sensor's paired φ/φ+180 arcs are modelled as evenly spaced single arcs.

## Pseudo-real generation (`r2flow/utils/guidance.py`, `tools/make_pseudoreal.py`)

This is DRUM translated to a rectified flow (t=0 noise, t=1 data):

|DRUM (VP diffusion)|here (rectified flow)|
|:-|:-|
|Tweedie x̂₀|`x1_hat = x_t + (1−t)·v(t, x_t)`|
|Π-GDM guidance, gated by raydrop mask `m`|`x1_hat += lam · m ⊙ (y − x1_hat)`, then step with the implied velocity|
|SDEdit init at t_init = 0.8|start at `t0 = 0.2` from `0.2·y + 0.8·ε`|
|RePaint, 3 cycles|`--resample` re-noise/repeat passes per step|
|η = −0.3 on DRUM's range encoding|a pixel is a return iff it decodes inside [min_depth, max_depth]|

DRUM only ever *removes* points. Our real data also *adds* them (ghosts, multipath, body hits), so
`--mask-mode ghost` additionally leaves the prior free in two cases:

- where the sim pixel is empty;
- where the prior predicts a return more than `ghost_rel` (25%) nearer than the sim surface.

Each sample is written in three compositions:

|`compose`|range on kept pixels|drops|added (ghost) points|
|:-|:-|:-|:-|
|`sim-range`|sim (DRUM exactly)|✓|—|
|`ghost` (default for training)|sim|✓|✓ generated|
|`full`|generated (keeps range noise)|✓|✓|

`stats.json` records the kept, added and dropped fractions. Figures go under `figures/`.

## Conditional model

- `--cond_channels 1` concatenates the pseudo-real image to `x_t`.
- `load_pretrained_weights` zero-initialises the new tokenizer input channel, so training starts
  exactly at the sim prior.
- `--augment True` applies the same random azimuth roll and flip to target and condition.
- Two sources are trained:
  - `--source noise`: `x_0 ~ N(0, I)`, the standard conditional flow.
  - `--source bridge`: `x_0 = cond + 0.1·ε`, which starts near the answer and is meant for 1–4 step
    live inference.

## Evaluation (`tools/eval_real2sim.py`)

**Paired.** Uses pseudo-real pairs from the held-out sim episodes (ep008/ep009), so ground truth
exists. Metrics:
- AbsRel and MAE on pixels both have;
- fill error;
- ghost rate: predicted returns where the GT is empty or >25% farther;
- miss rate;
- Chamfer distance.

The identity row (prediction = pseudo-real input) is the baseline the model has to beat.

**Unpaired.** Uses the real bag's held-out tail. The fill, per-row fill profile and log-depth
histogram (JSD) of the raw real scans and of their translations are compared against sim test
scans, alongside figures. The real scans are a different building from the sim, so this checks the
distribution only, not per-scan correctness.

## Deployment (`tools/real2sim_node.py`)

```bash
# ROS 2 (one environment needs rclpy + torch + natten)
python tools/real2sim_node.py ros --ckpt <cond .pth> --steps 4 --bf16
# offline: same code path, writes /real2sim/{input,cloud} to a bag for RViz + latency json
python tools/real2sim_node.py bag --ckpt <cond .pth> --steps 4 --bf16 \
    --bag r2flow/data/rosbags/go2_mocam_loop --start-message 2712 --out-bag logs/replay
```

The node runs the same preprocessing as training:
- 8-message window, stride 2;
- z-flip, then projection with `tools/lidar_grid.RayGrid`;
- output unprojected and z-flipped back into `go2/utlidar_lidar`.

Note that on this machine the `ros2_humble` env (Python 3.14) has no torch and `r2sflow` has no
rclpy. The ROS front end therefore needs a combined environment on the robot or the offboard
machine. The bag mode covers testing until then.

## Caveats

- **Scene mismatch.** The real prior is one 220 s non-hospital trajectory, so it learns L2 *sensor*
  artifacts in a different scene. The SDEdit start and guidance are what keep hospital content.
  Swap in real hospital bags (same converter flags) when they exist.
- **No de-skewing.** The real sweeps are not de-skewed because the bag has no /tf. The live node
  matches that.
- **Depth only.** The sim has no intensity, so reflectance is ignored.

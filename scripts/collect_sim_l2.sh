#!/usr/bin/env bash
# Collect hospital-sim episodes with the Unitree-L2 lidar emulation
# (../lidargen, lidar_config=Unitree_L2) as a Slurm array: one episode per task,
# each with its own spawn seed and ROS_DOMAIN_ID so tasks sharing a node do not
# hear each other.
#
#   EPISODES=10 bash scripts/collect_sim_l2.sh
#   EPISODES=1 GOALS=1 TIMEOUT=600 OUT=r2flow/data/rosbags/l2_smoke bash scripts/collect_sim_l2.sh
#
# Bags land in $OUT/hospital_epXXX_*/bag with /lidar_points (sensor frame, like the
# real /go2/raw_lidar) next to the usual /registered_scan_full.
set -euo pipefail
cd "$(dirname "$0")/.."
R2S=$(pwd)
LIDARGEN=${LIDARGEN:-$(cd ../lidargen && pwd)}
EPISODES=${EPISODES:-10}
GOALS=${GOALS:-5}
TIMEOUT=${TIMEOUT:-1500}
BASE_SEED=${BASE_SEED:-100}
OUT=${OUT:-r2flow/data/rosbags/simulated_bags_l2}
EXCLUDE=${EXCLUDE:-ryu,adrian,zac,jinkazama,kenmasters}
mkdir -p "$OUT" logs/l2sim
OUT=$(cd "$OUT" && pwd)

sbatch --parsable --job-name=l2-collect --partition=debug --exclude="$EXCLUDE" \
    --array=0-$((EPISODES - 1)) --gres=gpu:nvidia:1 --cpus-per-task=12 --mem=48G --time=1:30:00 \
    --output="${R2S}/logs/l2sim/collect-%A_%a.out" \
    --wrap="set -eo pipefail; export ROS_DOMAIN_ID=\$((20 + SLURM_ARRAY_TASK_ID % 200)) OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y; \
cd ${LIDARGEN}; nvidia-smi --query-gpu=name --format=csv,noheader || true; \
python3 src/lidargen/scripts/collect_data.py --scenes hospital --episodes-per-scene 1 \
  --goals ${GOALS} --episode-timeout ${TIMEOUT} --stack-ready-timeout 300 --base-seed \$((${BASE_SEED} + SLURM_ARRAY_TASK_ID)) --out ${OUT}"

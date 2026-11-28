#!/usr/bin/env bash
# Re-render the recorded hospital-sim trajectories with the Unitree-L2 lidar
# emulation (../lidargen), one Slurm array task per episode, then convert them
# to the go2w_l2_sim dataset. The trajectories (robot poses) are identical to
# go2w_nav_sim / go2w_common_sim; only the lidar model changes.
#
#   bash scripts/replay_sim_l2.sh          # prints the job ids: replay array, convert
#
# Needs poses first (CPU, seconds):
#   python tools/extract_sim_poses.py --episodes-dir r2flow/data/rosbags/simulated_bags \
#       --out r2flow/data/rosbags/simulated_l2_replay
set -euo pipefail
cd "$(dirname "$0")/.."
R2S=$(pwd)
LIDARGEN=${LIDARGEN:-$(cd ../lidargen && pwd)}
EPDIR=${EPDIR:-r2flow/data/rosbags/simulated_l2_replay}
EXCLUDE=${EXCLUDE:-ryu,adrian,zac,jinkazama,kenmasters}
mkdir -p logs/l2sim
EPDIR=$(cd "$EPDIR" && pwd)
mapfile -t EPS < <(ls -d "$EPDIR"/*/ | xargs -n1 basename)
N=${#EPS[@]}
printf '%s\n' "${EPS[@]}" > "$EPDIR/episodes.txt"

REPLAY=$(sbatch --parsable --job-name=l2-replay --partition=debug --exclude="$EXCLUDE" \
    --array=0-$((N - 1)) --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=48G --time=2:00:00 \
    --output="${R2S}/logs/l2sim/replay-%A_%a.out" \
    --wrap="eval \"\$(conda shell.bash hook)\"; conda activate lidargen; set -euo pipefail; \
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y; nvidia-smi --query-gpu=name --format=csv,noheader || true; \
EP=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${EPDIR}/episodes.txt); cd ${LIDARGEN}; \
python -m lidargen.scripts.replay_poses_l2 --episode ${EPDIR}/\$EP --seed \$SLURM_ARRAY_TASK_ID")

CONVERT=$(sbatch --parsable --job-name=l2-convert --partition=debug --exclude="$EXCLUDE" \
    --dependency=afterok:${REPLAY} --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${R2S}/logs/l2sim/convert-%j.out" \
    --wrap="eval \"\$(conda shell.bash hook)\"; conda activate r2sflow; set -euo pipefail; cd ${R2S}; \
rm -rf r2flow/data/go2w_l2_sim/dataset; \
python tools/sim_l2_to_r2flow.py --episodes-dir ${EPDIR} --out r2flow/data/go2w_l2_sim/dataset")
echo "replay=${REPLAY} convert=${CONVERT}"

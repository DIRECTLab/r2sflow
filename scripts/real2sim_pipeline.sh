#!/usr/bin/env bash
# End-to-end real->sim pipeline (DRUM-style pseudo-real pairs), as Slurm jobs
# chained with --dependency=afterok. See docs/REAL2SIM.md.
#
#   bash scripts/real2sim_pipeline.sh            # full run
#   SMOKE=1 bash scripts/real2sim_pipeline.sh    # tiny settings, catches bugs in minutes
#   SIM_DATASET=go2w_l2_sim REAL_FROM=logs/real2sim/<run> bash scripts/real2sim_pipeline.sh
#       # sim scans from the L2-emulating sim (tools/sim_l2_to_r2flow.py), reusing an
#       # earlier run's real prior (the real data is unchanged)
#   AFTER=<job id> ...   # hold the whole chain until that job (e.g. a dataset build) succeeds
#   RUN=<existing run> bash scripts/real2sim_pipeline.sh   # resume: stages whose
#       outputs exist are skipped (prep/priors by checkpoint; pseudo-real skips
#       already-written pairs), everything downstream is resubmitted
#
# Stages:
#   prep        build the HF caches for go2_common_real / go2w_common_sim (CPU)
#   real_prior  unconditional R2Flow on real Go2 sweeps     (KITTI-360 init)
#   sim_prior   unconditional R2Flow on hospital sim         (KITTI-360 init)
#   pseudoreal  sharded guided sampling sim -> pseudo-real   (needs real_prior)
#   merge       frames.json / stats.json for the pair dirs
#   cond_*      conditional real->sim models                 (sim_prior init)
#   eval_*      paired + unpaired metrics, figures
#   replay_*    run the model over the real bag's held-out tail, write a bag + latency
#
# Prerequisite: the real sweeps converted onto the common grid (CPU, ~3 min):
#   python tools/rosbag2_go2_to_r2flow.py --bag r2flow/data/rosbags/go2_mocam_loop \
#       --out r2flow/data/go2_common_real/dataset --height 64 --width 1024 \
#       --h-up 0 --h-down -24 --min-depth 0.5 --max-depth 30 --messages-per-scan 8 --stride 2
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-0}
if [ -z "${RUN:-}" ]; then
    RUN=$(date +%Y%m%dT%H%M%S)
    [ "$SMOKE" = 1 ] && RUN="smoke-${RUN}"
fi
case "$RUN" in smoke-*) SMOKE=1 ;; esac
# Nodes that do not mount this repo (jobs there fail instantly) or are not NVIDIA.
EXCLUDE=${EXCLUDE:-ryu,adrian,zac,jinkazama,kenmasters}
if [ "$SMOKE" = 1 ]; then
    PRIOR_IMAGES=1600; COND_IMAGES=1600; WARMUP=160; CKPT_EVERY=100; LOG_EVERY=100
    PR_ARGS="--limit 8 --seeds 2 --steps 8 --resample 1 --figures 4"; NUM_SHARDS=2
    EVAL_ARGS="--max-pairs 16 --max-real 16 --steps 1 4 --figures 4"; REPLAY_LIMIT=10
else
    PRIOR_IMAGES=200000; COND_IMAGES=400000; WARMUP=8000; CKPT_EVERY=5000; LOG_EVERY=2500
    PR_ARGS="--seeds 4 --steps 32 --resample 1 --figures 24"; NUM_SHARDS=4
    EVAL_ARGS="--max-pairs 512 --steps 1 2 4 8 32 --figures 16"; REPLAY_LIMIT=0
fi
# Sim side: go2w_common_sim (rotary preset, re-projected) or go2w_l2_sim (L2 emulation).
SIM_DATASET=${SIM_DATASET:-go2w_common_sim}
OUT=logs/real2sim/${RUN}
PAIRS=r2flow/data/go2w_pseudoreal/${RUN}
mkdir -p "${OUT}/slurm"
echo "run ${RUN}: sim ${SIM_DATASET}, logs in ${OUT}, pairs in ${PAIRS}"
echo "${SIM_DATASET}" > "${OUT}/sim_dataset"
# Reuse another run's real prior (same real data) instead of retraining it.
if [ -n "${REAL_FROM:-}" ] && [ ! -e "${OUT}/real_prior" ]; then
    ln -s "$(cd "${REAL_FROM}/real_prior" && pwd)" "${OUT}/real_prior"
fi

PRE='eval "$("${CONDA_EXE:-conda}" shell.bash hook)"; conda activate r2sflow; set -euo pipefail; nvidia-smi --query-gpu=name --format=csv,noheader || true;'
# Newest checkpoint under a train.py output dir.
LATEST='latest() { ls -t "$1"/*/spherical-64x1024/*/models/checkpoint_*.pth | head -1; };'
have_ckpt() { compgen -G "$1/*/spherical-64x1024/*/models/checkpoint_*.pth" >/dev/null; }
dep() {  # dep <job ids...>: an afterok clause over the ids that were submitted
    local ids=() i
    for i in "$@"; do [ -n "$i" ] && ids+=("$i"); done
    if [ ${#ids[@]} -gt 0 ]; then echo "--dependency=afterok:$(IFS=:; echo "${ids[*]}")"; fi
}

submit() {  # submit <name> <sbatch args...> -- <command>
    local name=$1; shift
    local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    sbatch --parsable --job-name="r2s-${name}" --partition=debug --exclude="${EXCLUDE}" \
        --output="${OUT}/slurm/${name}-%A_%a.out" "${opts[@]}" \
        --wrap="${PRE} ${LATEST} $*"
}

TRAIN_COMMON="--resolution 64 1024 --projection spherical-64x1024 --min_depth 0.5 --max_depth 30.0 \
 --train_reflectance False --init_weights r2flow-kitti360-1rf --lr 2e-5 --augment True \
 --num_images_lr_warmup ${WARMUP} --num_steps_checkpoint ${CKPT_EVERY} --num_steps_logging ${LOG_EVERY}"
ONE_GPU="accelerate launch --num_processes 1"
TWO_GPU="accelerate launch --multi_gpu --num_processes 2 --num_machines 1"

PREP= REAL= SIM=
if ! have_ckpt "${OUT}/real_prior" || ! have_ckpt "${OUT}/sim_prior"; then
PREP=$(submit prep $(dep ${AFTER:-}) --cpus-per-task=16 --mem=48G --time=2:00:00 -- \
  "python -c \"import datasets as ds
for n in ['go2_common_real', '${SIM_DATASET}']:
    for s in ['train', 'test']:
        d = ds.load_dataset(f'r2flow/data/{n}', name='spherical-64x1024', split=s, num_proc=16, trust_remote_code=True)
        m = d.with_format('numpy')[:256]['mask']
        print(n, s, len(d), 'mean fill', float(m.mean()))\"")
fi

have_ckpt "${OUT}/real_prior" || REAL=$(submit real_prior $(dep $PREP) --nodelist=bowser --gres=gpu:nvidia:2 \
  --cpus-per-task=16 --mem=48G --time=12:00:00 -- \
  "${TWO_GPU} train.py --dataset go2_common_real ${TRAIN_COMMON} --num_workers 8 \
   --num_images_training ${PRIOR_IMAGES} --output_dir ${OUT}/real_prior")

have_ckpt "${OUT}/sim_prior" || SIM=$(submit sim_prior $(dep $PREP) --nodelist=zengief --gres=gpu:nvidia:1 \
  --cpus-per-task=12 --mem=48G --time=12:00:00 -- \
  "${ONE_GPU} train.py --dataset ${SIM_DATASET} ${TRAIN_COMMON} --num_workers 8 \
   --num_images_training ${PRIOR_IMAGES} --output_dir ${OUT}/sim_prior")

# Needs the sim caches (PREP) even when the real prior is reused.
PR=$(submit pseudoreal $(dep $REAL $PREP) --array=0-$((NUM_SHARDS - 1)) \
  --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=32G --time=8:00:00 -- \
  "python tools/make_pseudoreal.py --ckpt \$(latest ${OUT}/real_prior) --out ${PAIRS} \
   --sim-dataset ${SIM_DATASET} --shard \${SLURM_ARRAY_TASK_ID} --num-shards ${NUM_SHARDS} --bf16 ${PR_ARGS}")

MERGE=$(submit merge $(dep $PR) --cpus-per-task=2 --mem=8G --time=0:30:00 -- \
  "python tools/make_pseudoreal.py --out ${PAIRS} --merge --sim-dataset ${SIM_DATASET}")

# Main model: ghost composition, noise source (bowser). Few-step variant: bridge source (zengief).
COND_COMMON="--dataset go2w_pseudoreal --data_dir ${PAIRS}/ghost --cond_channels 1 \
 --resolution 64 1024 --projection spherical-64x1024 --min_depth 0.5 --max_depth 30.0 \
 --train_reflectance False --lr 2e-5 --augment True --num_images_training ${COND_IMAGES} \
 --num_images_lr_warmup ${WARMUP} --num_steps_checkpoint ${CKPT_EVERY} --num_steps_logging ${LOG_EVERY}"

CN=$(submit cond_noise $(dep $MERGE $SIM) --nodelist=bowser --gres=gpu:nvidia:2 \
  --cpus-per-task=16 --mem=48G --time=12:00:00 -- \
  "${TWO_GPU} train.py ${COND_COMMON} --init_weights \$(latest ${OUT}/sim_prior) --source noise \
   --num_workers 8 --output_dir ${OUT}/cond_noise")
CB=$(submit cond_bridge $(dep $MERGE $SIM) --nodelist=zengief --gres=gpu:nvidia:1 \
  --cpus-per-task=12 --mem=48G --time=12:00:00 -- \
  "${ONE_GPU} train.py ${COND_COMMON} --init_weights \$(latest ${OUT}/sim_prior) --source bridge \
   --bridge_sigma 0.1 --num_workers 8 --output_dir ${OUT}/cond_bridge")

for V in noise bridge; do
    DEP=$([ $V = noise ] && echo $CN || echo $CB)
    submit eval_${V} $(dep $DEP) --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=32G --time=4:00:00 -- \
      "python tools/eval_real2sim.py --ckpt \$(latest ${OUT}/cond_${V}) --pairs ${PAIRS}/ghost \
       --sim ${SIM_DATASET} --bf16 ${EVAL_ARGS} --out ${OUT}/eval_${V}" >/dev/null
    # Held-out tail of the bag: the last 20% of sweeps start at message ~2712 (stride 2).
    submit replay_${V} $(dep $DEP) --gres=gpu:nvidia:1 --cpus-per-task=8 --mem=32G --time=2:00:00 -- \
      "python tools/real2sim_node.py bag --ckpt \$(latest ${OUT}/cond_${V}) --steps 4 --bf16 \
       --bag r2flow/data/rosbags/go2_mocam_loop --start-message 2712 --limit ${REPLAY_LIMIT} \
       --out-bag ${OUT}/replay_${V}" >/dev/null
done

echo "submitted: prep=${PREP:-skip} real=${REAL:-skip} sim=${SIM:-skip} pseudoreal=${PR} merge=${MERGE} cond_noise=${CN} cond_bridge=${CB}"
squeue -u "$USER" -o "%.10i %.22j %.8T %.10M %.20R %E" | grep -E "JOBID|r2s-" || true

#!/usr/bin/env bash
# SLURM launcher for the mjai-ach Ray cluster (Phase 3, AGENTS.md §1 D2).
#
# Submitted on a Linux SLURM cluster; spawns a Ray head + N workers across the
# allocated nodes, then runs the experiment config passed as $1. Phase-1 code
# is written so that this is the only Phase-3 addition (everything else is
# config + tuning per AGENTS.md).
#
# Usage on a SLURM cluster:
#   sbatch scripts/slurm/launch_ray_cluster.sh configs/exp/kuhn_ach_mirror.yaml
#
# This file is committed in Phase 1 (Step 8) but only RUN once a Linux+SLURM
# cluster exists; on Windows/home-CPU Phase-1 dev it is not exercised.

#SBATCH --job-name=mjai-ach
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

CONFIG="${1:-configs/exp/kuhn_ach_mirror.yaml}"
# Ray ports (head + dashboard + client). Adjust if your cluster reserves these.
RAY_HEAD_PORT=6379
RAY_DASHBOARD_PORT=8265

module load cuda/12.8 2>/dev/null || true  # adjust to your cluster's module name

# --- Phase 1 code: `uv sync` to materialize the env on each node ---
uv sync --extra dev

# --- Head node: start the Ray head ---
HEAD_NODE=$(hostname)
echo "[$(date)] Starting Ray head on $HEAD_NODE"
ray start --head --port="$RAY_HEAD_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
    --num-gpus="${SLURM_GPUS_PER_TASK:-4}" \
    --block=false &
HEAD_PID=$!
sleep 30  # let the head come up before workers connect
RAY_ADDRESS="$HEAD_NODE:$RAY_HEAD_PORT"
echo "[$(date)] Ray head address: $RAY_ADDRESS"

# --- Worker nodes: start Ray workers pointed at the head ---
if [[ "$SLURM_NODEID" -ne 0 ]]; then
    echo "[$(date)] Starting Ray worker on $(hostname)"
    ray start --address="$RAY_ADDRESS" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
        --num-gpus="${SLURM_GPUS_PER_TASK:-4}" \
        --block=false
fi

# --- Head runs the experiment, connecting to the auto-started cluster ---
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    echo "[$(date)] Head launching training: $CONFIG"
    # ray.init(address="auto") in the runner picks up the cluster we just built.
    MJAI_RAY_ADDRESS="auto" uv run mjai-train --config "$CONFIG"
    echo "[$(date)] Training complete; shutting down Ray"
    ray stop
    kill "$HEAD_PID" 2>/dev/null || true
fi

wait

#!/usr/bin/env bash
# All tasks in ONE job (clusters that limit jobs per user), JOBS at a time = CPUs requested:
#   sbatch slurm/run_all.sbatch
#   sbatch --export=ALL,OUT=/scratch/$USER/demos slurm/run_all.sbatch
# Adjust partition/account. Resubmitting continues where the job stopped (.done markers).
# Measured 9.25 on gpu-4080-402 (RTX 4080 SUPER): rendering adds 1-2 ms per step.
#SBATCH --job-name=demogen-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-4}} ./run_all.sh

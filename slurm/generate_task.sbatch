#!/usr/bin/env bash
# One task per job:
#   sbatch --export=TASK=pickcube slurm/generate_task.sbatch
#   for t in pickcube stackcube pushcube pullcube peginsertionside plugcharger; do
#     sbatch --job-name=gen-$t --export=TASK=$t slurm/generate_task.sbatch; done
# Extra generate_task.py arguments: --export=TASK=plugcharger,ARGS="--control-mode pd_joint_pos"
# Adjust partition/account and the GPU line to the cluster. The GPU is only for Vulkan
# rendering; drop it if the nodes render with lavapipe. Resubmitting continues where a
# job stopped (finished steps are marked .done).
#SBATCH --job-name=demogen
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TASK:?pass --export=TASK=<task>}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
unset UV_PROJECT_ENVIRONMENT
# shellcheck disable=SC2086  # ARGS is a list of arguments
.venv/bin/python generate_task.py "$TASK" ${ARGS:-}

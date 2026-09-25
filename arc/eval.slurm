#!/bin/bash
# =============================================================================
# eval.slurm -- checkpoint sweep: metrics table + route choice over training.
# =============================================================================
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/eval.slurm \
#          --run results/runs/a_s42 --baselines random scripted_fast scripted_slow
# Output: results/eval/<run>/{table.md, checkpoints.csv, episodes.jsonl, curves.png}
# =============================================================================
#SBATCH --job-name=drone-eval
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[drone] START eval ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" eval/evaluate.py --headless "$@"

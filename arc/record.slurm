#!/bin/bash
# =============================================================================
# record.slurm -- early vs final flythrough videos (median-representative take).
# =============================================================================
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/record.slurm \
#          --run results/runs/a_s42 --checkpoints 250 2000
# Output: results/videos/<run>/iter_<N>/{take_*.mp4, representative.mp4/.gif, manifest.json}
# =============================================================================
#SBATCH --job-name=drone-record
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[drone] START record ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" eval/record_video.py --headless "$@"

#!/bin/bash
# =============================================================================
# train.slurm -- one PPO run on 1 x L40S (Falcon). Submit FROM THE REPO ROOT.
# =============================================================================
# Everything after the script name goes to training/train.py's argparse, so a
# typo fails loudly (`VAR=x sbatch` is silently dropped -- playbook s2).
#
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/train.slurm \
#          --run-name a_s42 --resume auto
#   sbatch ... arc/train.slurm --run-name b_s42 --resume auto \
#          --set maze.config=configs/mazes/maze_b_overpass.yaml
#   # 3 seeds in parallel:
#   sbatch --array=0-2 ... arc/train.slurm --run-name a --seed-from-array --resume auto
#
# Walltime kill => resubmit the IDENTICAL line; --resume auto + fixed --run-name
# continues from the last model_<N>.pt (numeric order) under the same W&B run.
# L40S is required, not preferred: the policy's camera needs RT cores.
# Run arc/validate.slurm for the maze FIRST -- its throughput number sizes --time.
# =============================================================================
#SBATCH --job-name=drone-train
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# --account / --mail-user are NOT here: public repo; pass them on the command line.
# --mem explicit: --mem=0 (whole node) never backfills.

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[drone] START train ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" training/train.py --config configs/train.yaml --headless "$@"

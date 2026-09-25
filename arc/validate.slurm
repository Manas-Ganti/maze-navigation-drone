#!/bin/bash
# =============================================================================
# validate.slurm -- the in-sim section 6 checklist + Phase 0 probes, then throughput.
# =============================================================================
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/validate.slurm \
#          --set maze.config=configs/mazes/maze_a_window.yaml
#
# Output: results/validate_sim/<maze>/report_checks.json, report_throughput.json,
#         frames_grid.png (look at it: the 6.6 human check).
# Do not submit training for a maze until this job prints ALL PASS.
# =============================================================================
#SBATCH --job-name=drone-validate
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[drone] START validate ${SLURM_JOB_ID} on $(hostname): $*"

# pure half first (seconds): a failing maze should not reach the GPU checks
"$PY" tools/check_mazes.py "$@" --no-plot || { echo "pure maze checks FAILED"; exit 1; }
rc=0
"$PY" eval/validate_sim.py --headless --mode checks "$@" || rc=$?
# throughput runs even if a check failed: the number is needed to size --time
"$PY" eval/validate_sim.py --headless --mode throughput "$@" || true
exit $rc

#!/bin/bash
#SBATCH --partition=YOUR_PARTITION
#SBATCH --job-name=arm64-swebench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --exclude=lego-cg1-qs-89
#SBATCH --output=slurm_logs/arm_build_%j.out
#SBATCH --error=slurm_logs/arm_build_%j.err

# Usage:
#   sbatch launch_arm_build.slurm                  # full build + push
#   sbatch launch_arm_build.slurm --retry_failed   # retry only previously failed instances
#   sbatch launch_arm_build.slurm --env_only       # build env images only (for debugging)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
NUM_WORKERS=8
STATE_FILE="${SCRIPT_DIR}/build_state.json"
REGISTRY="${REGISTRY:-registry.example.com/namespace/swe-bench}"
VENV_DIR="${SCRIPT_DIR}/.venv"
EXTRA_ARGS=("$@")
# ──────────────────────────────────────────────────────────────

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Workers:       ${NUM_WORKERS}"
echo "State file:    ${STATE_FILE}"
echo "Registry:      ${REGISTRY}"
echo "Extra args:    ${EXTRA_ARGS[*]:-<none>}"
echo "Start time:    $(date)"
echo "============================================"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

# ── Activate virtual environment ──
if [ -d "${VENV_DIR}" ]; then
    echo "Activating venv at ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
else
    echo "WARNING: No venv found at ${VENV_DIR}, using system python3"
fi

echo "Python: $(which python3) ($(python3 --version))"
echo "============================================"

python3 -m swebench.harness.prepare_images \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --arch arm64 \
    --max_workers "${NUM_WORKERS}" \
    --state_file "${STATE_FILE}" \
    --registry "${REGISTRY}" \
    "${EXTRA_ARGS[@]}"

python -m swebench.harness.prepare_images \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids_file fixable_instances.txt \
  --arch arm64 \
  --max_workers 4 \
  --state_file build_state.json \
  --registry "${REGISTRY}" \
  --verify true \
  --force_rebuild true

EXIT_CODE=$?

echo "============================================"
echo "Job ${SLURM_JOB_ID} finished with exit code ${EXIT_CODE}"
echo "End time: $(date)"
echo "============================================"

exit ${EXIT_CODE}

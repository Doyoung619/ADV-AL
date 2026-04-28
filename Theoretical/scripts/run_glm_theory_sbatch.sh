#!/usr/bin/env bash
# GLM theory experiment launcher — SLURM array + flock work-stealing queue.
#
# Pattern adopted from Empirical/scripts/run_experimentF_badge_secant_prefilter_s0_4_sbatch.sh.
#
# Usage:
#   bash Theoretical/scripts/run_glm_theory_sbatch.sh --dry-run     # write all scripts, do not submit
#   bash Theoretical/scripts/run_glm_theory_sbatch.sh --submit      # write + submit array + aggregator
#
# What it submits:
#   1) An sbatch array job sized to MAX_GPUS (default 8). Each array task pulls (GLM, seed)
#      jobs from a shared TSV via flock and runs them with PROCS_PER_GPU workers per task.
#   2) A dependent aggregator job that runs after the array job succeeds. It merges all
#      per-task metrics.csv files and writes outputs into:
#          experiments/experimentG, experimentH, experimentI
#      with figures/ subfolders inside each.
#
# Knobs (all overridable as env vars):
#   MAX_GPUS=8          — array size; cluster cap is 8 without -q big_qos
#   PROCS_PER_GPU=2     — workers per GPU slot (each runs one (GLM, seed))
#   SEEDS=0,1,2,3,4
#   GLMS=gaussian,logistic,softmax,poisson
#   ROUNDS=20  BATCH_SIZE=50  POOL_SIZE=2000  INIT_SIZE=100  DIM=20  NUM_CLASSES=5
#   ALPHA=75  EPS_ACQ=0.25
#   CONTRACTION_K=100   CONTRACTION_LR=0.01
#   PARTITION=          — leave empty for default partition
#   CONDA_ENV=doyoung_env
#   CPUS_PER_TASK=8  MEM_PER_GPU=30G  TIME_LIMIT=06:00:00

set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--submit" ]]; then
  echo "Usage: bash $0 [--dry-run|--submit]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THEORETICAL_DIR="${ROOT_DIR}/Theoretical"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"
STAGING_ROOT="${STAGING_ROOT:-${EXPERIMENTS_ROOT}/_glm_theory_runs}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${EXPERIMENTS_ROOT}/_launch_glm_theory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-doyoung_env}"

SEEDS="${SEEDS:-0,1,2,3,4}"
GLMS="${GLMS:-gaussian,logistic,softmax,poisson}"
METHODS="${METHODS:-random badge secant_badge}"

ROUNDS="${ROUNDS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
POOL_SIZE="${POOL_SIZE:-2000}"
INIT_SIZE="${INIT_SIZE:-100}"
DIM="${DIM:-20}"
NUM_CLASSES="${NUM_CLASSES:-5}"
ALPHA="${ALPHA:-75}"
EPS_ACQ="${EPS_ACQ:-0.25}"
CONTRACTION_K="${CONTRACTION_K:-100}"
CONTRACTION_LR="${CONTRACTION_LR:-0.01}"

MAX_GPUS="${MAX_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
PARTITION="${PARTITION:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-30G}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"

if (( MAX_GPUS < 1 || MAX_GPUS > 8 )); then
  echo "MAX_GPUS must be in [1,8] (got ${MAX_GPUS})" >&2
  exit 2
fi
if (( PROCS_PER_GPU < 1 )); then
  echo "PROCS_PER_GPU must be >= 1 (got ${PROCS_PER_GPU})" >&2
  exit 2
fi

mkdir -p "${STAGING_ROOT}" "${LAUNCH_ROOT}/logs" "${LAUNCH_ROOT}/slurm" "${LAUNCH_ROOT}/state"

COMMANDS_TSV="${LAUNCH_ROOT}/commands.tsv"
: > "${COMMANDS_TSV}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a GLM_ARRAY  <<< "${GLMS}"

job_idx=0
for glm in "${GLM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    out_dir="${STAGING_ROOT}/${glm}_seed${seed}"
    log_path="${LAUNCH_ROOT}/logs/${glm}_seed${seed}.log"
    cmd="${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/run_glm_theory_experiments.py \
--glm ${glm} \
--seeds ${seed} \
--methods ${METHODS} \
--rounds ${ROUNDS} \
--batch-size ${BATCH_SIZE} \
--pool-size ${POOL_SIZE} \
--init-size ${INIT_SIZE} \
--dim ${DIM} \
--num-classes ${NUM_CLASSES} \
--alpha ${ALPHA} \
--eps-acq ${EPS_ACQ} \
--contraction-K ${CONTRACTION_K} \
--contraction-lr ${CONTRACTION_LR} \
--decay-curves-seed 0 \
--outdir ${out_dir} \
--skip-figures"
    printf '%s\t%s\t%s\t%s\n' \
      "${job_idx}" "${glm}_seed${seed}" "${log_path}" "${cmd}" >> "${COMMANDS_TSV}"
    job_idx=$((job_idx + 1))
  done
done

# ---------------------------------------------------------------------------
# worker.sh — pulls jobs from COMMANDS_TSV using flock, runs PROCS_PER_GPU in parallel.
# ---------------------------------------------------------------------------
cat > "${LAUNCH_ROOT}/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: worker.sh COMMANDS_TSV STATE_DIR PROCS_PER_GPU CONDA_ENV ROOT_DIR" >&2
  exit 2
fi

COMMANDS_TSV="$1"
STATE_DIR="$2"
PROCS_PER_GPU="$3"
CONDA_ENV="$4"
ROOT_DIR="$5"
LOCK_FILE="${STATE_DIR}/queue.lock"
CURSOR_FILE="${STATE_DIR}/cursor"
DONE_DIR="${STATE_DIR}/done"
mkdir -p "${DONE_DIR}"
touch "${CURSOR_FILE}" "${LOCK_FILE}"
if [[ ! -s "${CURSOR_FILE}" ]]; then
  echo 0 > "${CURSOR_FILE}"
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"

claim_next() {
  local line_count idx line
  exec 9>"${LOCK_FILE}"
  flock 9
  idx="$(cat "${CURSOR_FILE}")"
  line_count="$(wc -l < "${COMMANDS_TSV}")"
  if [[ "${idx}" -ge "${line_count}" ]]; then
    flock -u 9
    return 1
  fi
  line="$(sed -n "$((idx + 1))p" "${COMMANDS_TSV}")"
  echo $((idx + 1)) > "${CURSOR_FILE}"
  flock -u 9
  printf '%s\n' "${line}"
}

run_worker() {
  local worker_id="$1"
  local line job_idx tag log_path cmd status_file rc
  while line="$(claim_next)"; do
    IFS=$'\t' read -r job_idx tag log_path cmd <<< "${line}"
    status_file="${DONE_DIR}/${job_idx}_${tag}.status"
    if [[ -f "${status_file}" ]]; then
      continue
    fi
    mkdir -p "$(dirname "${log_path}")"
    {
      echo "[$(date '+%F %T')] START job=${job_idx} tag=${tag} host=$(hostname) array=${SLURM_ARRAY_TASK_ID:-na} worker=${worker_id} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
      echo "${cmd}"
    } >> "${log_path}"
    set +e
    bash -lc "${cmd}" >> "${log_path}" 2>&1
    rc="$?"
    set -e
    echo "[$(date '+%F %T')] END job=${job_idx} tag=${tag} rc=${rc}" >> "${log_path}"
    if [[ "${rc}" -eq 0 ]]; then
      echo "ok" > "${status_file}"
    else
      echo "failed:${rc}" > "${status_file}"
      exit "${rc}"
    fi
  done
}

pids=()
for ((worker = 0; worker < PROCS_PER_GPU; worker++)); do
  run_worker "${worker}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
WORKER
chmod +x "${LAUNCH_ROOT}/worker.sh"

# ---------------------------------------------------------------------------
# Array sbatch script
# ---------------------------------------------------------------------------
array_end=$((MAX_GPUS - 1))

partition_line=""
if [[ -n "${PARTITION}" ]]; then
  partition_line="#SBATCH -p ${PARTITION}"
fi

cat > "${LAUNCH_ROOT}/sbatch_array.sh" <<EOF
#!/bin/bash
#SBATCH -J glm_theory
${partition_line}
#SBATCH --array=0-${array_end}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LAUNCH_ROOT}/slurm/array_%A_%a.out
#SBATCH --error=${LAUNCH_ROOT}/slurm/array_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=${CPUS_PER_TASK}
export MKL_NUM_THREADS=${CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

bash "${LAUNCH_ROOT}/worker.sh" "${COMMANDS_TSV}" "${LAUNCH_ROOT}/state" "${PROCS_PER_GPU}" "${CONDA_ENV}" "${ROOT_DIR}"
EOF
chmod +x "${LAUNCH_ROOT}/sbatch_array.sh"

# ---------------------------------------------------------------------------
# Aggregator sbatch script — runs after the array job succeeds.
# ---------------------------------------------------------------------------
cat > "${LAUNCH_ROOT}/sbatch_aggregate.sh" <<EOF
#!/bin/bash
#SBATCH -J glm_theory_agg
${partition_line}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=${LAUNCH_ROOT}/slurm/aggregate_%j.out
#SBATCH --error=${LAUNCH_ROOT}/slurm/aggregate_%j.err

set -euo pipefail
eval "\$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"

${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/aggregate_and_plot.py \\
    --staging-dir ${STAGING_ROOT} \\
    --out-root ${EXPERIMENTS_ROOT} \\
    --glms ${GLM_ARRAY[*]}
EOF
chmod +x "${LAUNCH_ROOT}/sbatch_aggregate.sh"

# ---------------------------------------------------------------------------
# Submit script (one-shot): array first, aggregator depends on array success.
# ---------------------------------------------------------------------------
cat > "${LAUNCH_ROOT}/submit.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -rf "${LAUNCH_ROOT}/state"
mkdir -p "${LAUNCH_ROOT}/state"
ARRAY_JOB=\$(sbatch --parsable \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_array.sh")
echo "Array job: \${ARRAY_JOB}"
AGG_JOB=\$(sbatch --parsable --dependency=afterok:\${ARRAY_JOB} \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_aggregate.sh")
echo "Aggregator job: \${AGG_JOB} (after array \${ARRAY_JOB})"
echo
echo "Tail logs:  tail -F ${LAUNCH_ROOT}/logs/*.log"
echo "Watch:      squeue --me"
echo "Final out:  ${EXPERIMENTS_ROOT}/experimentG ; experimentH ; experimentI"
EOF
chmod +x "${LAUNCH_ROOT}/submit.sh"

# ---------------------------------------------------------------------------
echo "Wrote ${job_idx} commands -> ${COMMANDS_TSV}"
echo "Staging root  : ${STAGING_ROOT}"
echo "Launch root   : ${LAUNCH_ROOT}"
echo "Experiments   : ${EXPERIMENTS_ROOT}/{experimentG,experimentH,experimentI}"
echo "GPU array     : 0-${array_end} (${MAX_GPUS} slots)  PROCS_PER_GPU=${PROCS_PER_GPU}  → ${MAX_GPUS}*${PROCS_PER_GPU} parallel slots"
echo
echo "Submit with:"
echo "  bash ${LAUNCH_ROOT}/submit.sh"
echo
echo "One-shot:"
echo "  CONDA_ENV=${CONDA_ENV} MAX_GPUS=${MAX_GPUS} PROCS_PER_GPU=${PROCS_PER_GPU} bash $(realpath "${BASH_SOURCE[0]}") --submit"

if [[ "${MODE}" == "--submit" ]]; then
  bash "${LAUNCH_ROOT}/submit.sh"
fi

#!/usr/bin/env bash
# Long-horizon Experiment G (R=100) launcher.
#
# Re-uses the same per-task runner as the main launcher but:
#   - rounds = 100 (5x longer)
#   - pool   = 10000 (so each method has > 100 * batch picks available)
#   - aggregator runs with --only-experiments G --filename-suffix _R100, so the new
#     spectral-growth figures land in experiments/experimentG/figures/ alongside
#     the original R=20 PDFs without overwriting them.
#
# Usage:
#   bash Theoretical/scripts/run_glm_theory_G_R100_sbatch.sh --dry-run
#   bash Theoretical/scripts/run_glm_theory_G_R100_sbatch.sh --submit

set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--submit" ]]; then
  echo "Usage: bash $0 [--dry-run|--submit]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THEORETICAL_DIR="${ROOT_DIR}/Theoretical"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"
STAGING_ROOT="${STAGING_ROOT:-${EXPERIMENTS_ROOT}/_glm_theory_runs_R100}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${EXPERIMENTS_ROOT}/_launch_glm_theory_R100}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-doyoung_env}"

SEEDS="${SEEDS:-0,1,2,3,4}"
GLMS="${GLMS:-gaussian,logistic,softmax,poisson}"
METHODS="${METHODS:-random badge secant_badge}"

ROUNDS="${ROUNDS:-100}"
BATCH_SIZE="${BATCH_SIZE:-50}"
POOL_SIZE="${POOL_SIZE:-10000}"
INIT_SIZE="${INIT_SIZE:-100}"
DIM="${DIM:-20}"
NUM_CLASSES="${NUM_CLASSES:-5}"
ALPHA="${ALPHA:-75}"
EPS_ACQ="${EPS_ACQ:-0.25}"
CONTRACTION_K="${CONTRACTION_K:-100}"
CONTRACTION_LR="${CONTRACTION_LR:-0.01}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_R100}"

MAX_GPUS="${MAX_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
PARTITION="${PARTITION:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-30G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

if (( MAX_GPUS < 1 || MAX_GPUS > 8 )); then
  echo "MAX_GPUS must be in [1,8] (got ${MAX_GPUS})" >&2
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

cat > "${LAUNCH_ROOT}/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 5 ]]; then
  echo "Usage: worker.sh COMMANDS_TSV STATE_DIR PROCS_PER_GPU CONDA_ENV ROOT_DIR" >&2
  exit 2
fi
COMMANDS_TSV="$1"; STATE_DIR="$2"; PROCS_PER_GPU="$3"; CONDA_ENV="$4"; ROOT_DIR="$5"
LOCK_FILE="${STATE_DIR}/queue.lock"; CURSOR_FILE="${STATE_DIR}/cursor"; DONE_DIR="${STATE_DIR}/done"
mkdir -p "${DONE_DIR}"; touch "${CURSOR_FILE}" "${LOCK_FILE}"
[[ -s "${CURSOR_FILE}" ]] || echo 0 > "${CURSOR_FILE}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"
claim_next() {
  local line_count idx line
  exec 9>"${LOCK_FILE}"; flock 9
  idx="$(cat "${CURSOR_FILE}")"; line_count="$(wc -l < "${COMMANDS_TSV}")"
  if [[ "${idx}" -ge "${line_count}" ]]; then flock -u 9; return 1; fi
  line="$(sed -n "$((idx + 1))p" "${COMMANDS_TSV}")"
  echo $((idx + 1)) > "${CURSOR_FILE}"; flock -u 9
  printf '%s\n' "${line}"
}
run_worker() {
  local worker_id="$1" line job_idx tag log_path cmd status_file rc
  while line="$(claim_next)"; do
    IFS=$'\t' read -r job_idx tag log_path cmd <<< "${line}"
    status_file="${DONE_DIR}/${job_idx}_${tag}.status"
    [[ -f "${status_file}" ]] && continue
    mkdir -p "$(dirname "${log_path}")"
    {
      echo "[$(date '+%F %T')] START job=${job_idx} tag=${tag} host=$(hostname) array=${SLURM_ARRAY_TASK_ID:-na} worker=${worker_id} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
      echo "${cmd}"
    } >> "${log_path}"
    set +e; bash -lc "${cmd}" >> "${log_path}" 2>&1; rc="$?"; set -e
    echo "[$(date '+%F %T')] END job=${job_idx} tag=${tag} rc=${rc}" >> "${log_path}"
    if [[ "${rc}" -eq 0 ]]; then echo "ok" > "${status_file}"; else echo "failed:${rc}" > "${status_file}"; exit "${rc}"; fi
  done
}
pids=()
for ((worker = 0; worker < PROCS_PER_GPU; worker++)); do run_worker "${worker}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "${pid}"; done
WORKER
chmod +x "${LAUNCH_ROOT}/worker.sh"

array_end=$((MAX_GPUS - 1))
partition_line=""
[[ -n "${PARTITION}" ]] && partition_line="#SBATCH -p ${PARTITION}"

cat > "${LAUNCH_ROOT}/sbatch_array.sh" <<EOF
#!/bin/bash
#SBATCH -J glm_R100
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

cat > "${LAUNCH_ROOT}/sbatch_aggregate.sh" <<EOF
#!/bin/bash
#SBATCH -J glm_R100_agg
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
    --glms ${GLM_ARRAY[*]} \\
    --only-experiments G \\
    --filename-suffix ${FILENAME_SUFFIX}
EOF
chmod +x "${LAUNCH_ROOT}/sbatch_aggregate.sh"

cat > "${LAUNCH_ROOT}/submit.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -rf "${LAUNCH_ROOT}/state"
mkdir -p "${LAUNCH_ROOT}/state"
ARRAY_JOB=\$(sbatch --parsable \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_array.sh")
echo "Array job: \${ARRAY_JOB}"
AGG_JOB=\$(sbatch --parsable --dependency=afterok:\${ARRAY_JOB} \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_aggregate.sh")
echo "Aggregator job: \${AGG_JOB} (after array \${ARRAY_JOB})"
echo "Tail logs: tail -F ${LAUNCH_ROOT}/logs/*.log"
echo "Watch:     squeue --me"
echo "Output:    ${EXPERIMENTS_ROOT}/experimentG/figures/experiment_G_spectral_growth_*${FILENAME_SUFFIX}.pdf"
EOF
chmod +x "${LAUNCH_ROOT}/submit.sh"

echo "Wrote ${job_idx} commands -> ${COMMANDS_TSV}"
echo "Staging  : ${STAGING_ROOT}"
echo "Launch   : ${LAUNCH_ROOT}"
echo "Rounds=${ROUNDS}  Pool=${POOL_SIZE}  Suffix=${FILENAME_SUFFIX}"
echo "GPU array: 0-${array_end} (${MAX_GPUS} slots) PROCS_PER_GPU=${PROCS_PER_GPU} → $((MAX_GPUS * PROCS_PER_GPU)) parallel slots"
echo
echo "Submit:   bash ${LAUNCH_ROOT}/submit.sh"

[[ "${MODE}" == "--submit" ]] && bash "${LAUNCH_ROOT}/submit.sh"

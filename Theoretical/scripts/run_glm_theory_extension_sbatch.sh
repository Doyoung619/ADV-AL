#!/usr/bin/env bash
# Extension launcher: Exponential & Gamma GLMs for Experiments G and I.
#
# Submits per-(GLM, seed, horizon) tasks for {exponential, gamma} × {R=20, R=100} ×
# {seeds 0..4} = 20 jobs to the same flock-queue worker pool, then chains an
# aggregator that combines them with the existing R=20 (4-GLM) and R=100 (4-GLM)
# staging dirs to emit:
#   - experiment_G_spectral_growth_<glm>_R20.pdf  for {exponential, gamma}
#   - experiment_G_spectral_growth_<glm>_R100.pdf for {exponential, gamma}
#   - experiment_G_spectral_growth_<glm>_compare_R20_vs_R100.pdf for both
#   - experiment_I_lambdamin_vs_contraction_<glm>.pdf for both (R=20 data)
#   - experiment_I_lambdamin_vs_contraction_all_glms.pdf (combined, R=20 across 6 GLMs)
#
# By default the aggregator also waits on the existing R=100 (4 GLM) array job:
# pass DEPEND_ON_JOB=<jobid> to chain it; otherwise pass DEPEND_ON_JOB=skip.
#
# Usage:
#   bash Theoretical/scripts/run_glm_theory_extension_sbatch.sh --dry-run
#   bash Theoretical/scripts/run_glm_theory_extension_sbatch.sh --submit
#   DEPEND_ON_JOB=1408108 bash Theoretical/scripts/run_glm_theory_extension_sbatch.sh --submit

set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--submit" ]]; then
  echo "Usage: bash $0 [--dry-run|--submit]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THEORETICAL_DIR="${ROOT_DIR}/Theoretical"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"

# Existing staging dirs (already populated by previous runs).
STAGING_R20_4GLM="${STAGING_R20_4GLM:-${EXPERIMENTS_ROOT}/_glm_theory_runs}"
STAGING_R100_4GLM="${STAGING_R100_4GLM:-${EXPERIMENTS_ROOT}/_glm_theory_runs_R100}"

# New staging dirs for the exponential/gamma extension.
STAGING_EXT_R20="${STAGING_EXT_R20:-${EXPERIMENTS_ROOT}/_glm_theory_ext_R20}"
STAGING_EXT_R100="${STAGING_EXT_R100:-${EXPERIMENTS_ROOT}/_glm_theory_ext_R100}"

LAUNCH_ROOT="${LAUNCH_ROOT:-${EXPERIMENTS_ROOT}/_launch_glm_theory_extension}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-doyoung_env}"

SEEDS="${SEEDS:-0,1,2,3,4}"
EXT_GLMS="${EXT_GLMS:-exponential,gamma}"
METHODS="${METHODS:-random badge secant_badge}"

ROUNDS_SHORT="${ROUNDS_SHORT:-20}"
ROUNDS_LONG="${ROUNDS_LONG:-100}"
BATCH_SIZE="${BATCH_SIZE:-50}"
POOL_SIZE_SHORT="${POOL_SIZE_SHORT:-2000}"
POOL_SIZE_LONG="${POOL_SIZE_LONG:-10000}"
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
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

DEPEND_ON_JOB="${DEPEND_ON_JOB:-skip}"

if (( MAX_GPUS < 1 || MAX_GPUS > 8 )); then
  echo "MAX_GPUS must be in [1,8] (got ${MAX_GPUS})" >&2
  exit 2
fi

mkdir -p "${STAGING_EXT_R20}" "${STAGING_EXT_R100}" \
         "${LAUNCH_ROOT}/logs" "${LAUNCH_ROOT}/slurm" "${LAUNCH_ROOT}/state"

COMMANDS_TSV="${LAUNCH_ROOT}/commands.tsv"
: > "${COMMANDS_TSV}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a GLM_ARRAY  <<< "${EXT_GLMS}"

emit_task() {
  local glm="$1" seed="$2" rounds="$3" pool="$4" stage_dir="$5" tag_prefix="$6"
  local out_dir="${stage_dir}/${glm}_seed${seed}"
  local log_path="${LAUNCH_ROOT}/logs/${tag_prefix}_${glm}_seed${seed}.log"
  local cmd="${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/run_glm_theory_experiments.py \
--glm ${glm} \
--seeds ${seed} \
--methods ${METHODS} \
--rounds ${rounds} \
--batch-size ${BATCH_SIZE} \
--pool-size ${pool} \
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
  printf '%s\t%s\t%s\t%s\n' "${job_idx}" "${tag_prefix}_${glm}_seed${seed}" "${log_path}" "${cmd}" >> "${COMMANDS_TSV}"
  job_idx=$((job_idx + 1))
}

job_idx=0
for glm in "${GLM_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    emit_task "${glm}" "${seed}" "${ROUNDS_SHORT}" "${POOL_SIZE_SHORT}" "${STAGING_EXT_R20}"  "R20"
    emit_task "${glm}" "${seed}" "${ROUNDS_LONG}"  "${POOL_SIZE_LONG}"  "${STAGING_EXT_R100}" "R100"
  done
done

# worker.sh — same flock-based queue runner used by the main launcher.
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
#SBATCH -J glm_ext
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

# Aggregator: 3 invocations of aggregate_and_plot.py.
cat > "${LAUNCH_ROOT}/sbatch_aggregate.sh" <<EOF
#!/bin/bash
#SBATCH -J glm_ext_agg
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

ALL_DIRS="${STAGING_R20_4GLM} ${STAGING_R100_4GLM} ${STAGING_EXT_R20} ${STAGING_EXT_R100}"

# (1) R=20 base plots for the new GLMs.
${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/aggregate_and_plot.py \\
    --staging-dirs \${ALL_DIRS} \\
    --out-root ${EXPERIMENTS_ROOT} \\
    --glms exponential gamma \\
    --rounds-tag-filter ${ROUNDS_SHORT} \\
    --filename-suffix _R${ROUNDS_SHORT} \\
    --only-experiments G H I

# (2) R=100 base plots for the new GLMs (only G is asked for at long horizon, but we
#     also emit H/I for completeness).
${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/aggregate_and_plot.py \\
    --staging-dirs \${ALL_DIRS} \\
    --out-root ${EXPERIMENTS_ROOT} \\
    --glms exponential gamma \\
    --rounds-tag-filter ${ROUNDS_LONG} \\
    --filename-suffix _R${ROUNDS_LONG} \\
    --only-experiments G

# (3) Comparison + combined-I plots (use the full multi-horizon dataframe).
${PYTHON_BIN} -u ${THEORETICAL_DIR}/glm_theory/aggregate_and_plot.py \\
    --staging-dirs \${ALL_DIRS} \\
    --out-root ${EXPERIMENTS_ROOT} \\
    --glms gaussian logistic softmax poisson exponential gamma \\
    --skip-base-plots \\
    --compare-rounds-glms exponential gamma \\
    --combined-i-plot \\
    --only-experiments G I
EOF
chmod +x "${LAUNCH_ROOT}/sbatch_aggregate.sh"

cat > "${LAUNCH_ROOT}/submit.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -rf "${LAUNCH_ROOT}/state"
mkdir -p "${LAUNCH_ROOT}/state"
ARRAY_JOB=\$(sbatch --parsable \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_array.sh")
echo "Array job: \${ARRAY_JOB}"
DEP="afterok:\${ARRAY_JOB}"
if [[ "${DEPEND_ON_JOB}" != "skip" && -n "${DEPEND_ON_JOB}" ]]; then
  DEP="afterok:\${ARRAY_JOB}:${DEPEND_ON_JOB}"
  echo "Aggregator also waits on existing job ${DEPEND_ON_JOB}"
fi
AGG_JOB=\$(sbatch --parsable --dependency=\${DEP} \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_aggregate.sh")
echo "Aggregator job: \${AGG_JOB} (dep: \${DEP})"
echo "Tail logs: tail -F ${LAUNCH_ROOT}/logs/*.log"
echo "Watch:     squeue --me"
echo "Outputs into: ${EXPERIMENTS_ROOT}/experiment{G,H,I}/figures/"
EOF
chmod +x "${LAUNCH_ROOT}/submit.sh"

echo "Wrote ${job_idx} commands -> ${COMMANDS_TSV}"
echo "Existing staging dirs:"
echo "  R20  (4 GLM) : ${STAGING_R20_4GLM}"
echo "  R100 (4 GLM) : ${STAGING_R100_4GLM}"
echo "Extension staging dirs:"
echo "  R20  (exp,gamma) : ${STAGING_EXT_R20}"
echo "  R100 (exp,gamma) : ${STAGING_EXT_R100}"
echo "Launch  : ${LAUNCH_ROOT}"
echo "Array   : 0-${array_end}  PROCS_PER_GPU=${PROCS_PER_GPU}  → $((MAX_GPUS * PROCS_PER_GPU)) parallel slots"
echo
echo "Submit: bash ${LAUNCH_ROOT}/submit.sh"

[[ "${MODE}" == "--submit" ]] && bash "${LAUNCH_ROOT}/submit.sh"

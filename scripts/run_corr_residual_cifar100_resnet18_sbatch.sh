#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/experiments/set_ADV_main_resnet18_cifar100}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${EXPERIMENT_ROOT}/sbatch_corr_residual_p0_p10_p25_seed0_9}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-doyoung_env}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
GATES="${GATES:-0,10,25}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${LAUNCH_ROOT}/logs" "${LAUNCH_ROOT}/slurm"

COMMANDS_TSV="${LAUNCH_ROOT}/commands.tsv"
: > "${COMMANDS_TSV}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a GATE_ARRAY <<< "${GATES}"

job_idx=0
for gate in "${GATE_ARRAY[@]}"; do
  method="ours_corr_residual_refine"
  out_dir="${EXPERIMENT_ROOT}/cifar100_resnet18_ours_corr_residual_refine_p${gate}_b100_r20"
  mkdir -p "${out_dir}/logs"
  for seed in "${SEED_ARRAY[@]}"; do
    run_log="${out_dir}/logs/seed_${seed}.log"
    cmd="${PYTHON_BIN} ${ROOT_DIR}/main.py \
--dataset cifar100 \
--model resnet18 \
--acquisition-method ${method} \
--clean-gate-percentile ${gate} \
--initial_labeled_size 500 \
--acquisition_size 100 \
--num_rounds 20 \
--epochs_per_round 80 \
--epsilon 0.00392156862745098 \
--seed ${seed} \
--output-dir ${out_dir} \
--run-name seed_${seed} \
--train-mode adv \
--adv-train-attack pgd \
--adv-train-epsilon 0.00392156862745098 \
--adv-train-steps 3 \
--data-dir ${DATA_DIR} \
--num-workers ${NUM_WORKERS} \
--pin-memory \
--skip-logit-mismatch-eval \
--no-save-checkpoints"
    printf '%s\t%s\t%s\t%s\n' "${job_idx}" "p${gate}_seed${seed}" "${run_log}" "${cmd}" >> "${COMMANDS_TSV}"
    job_idx=$((job_idx + 1))
  done
done

cat > "${LAUNCH_ROOT}/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: worker.sh COMMANDS_TSV STATE_DIR PROCS_PER_GPU CONDA_ENV" >&2
  exit 2
fi

COMMANDS_TSV="$1"
STATE_DIR="$2"
PROCS_PER_GPU="$3"
CONDA_ENV="$4"
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
cd /lustre/wjwjwj/doyoung/ADV-AL

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
  local line job_idx tag log_path cmd status_file
  while line="$(claim_next)"; do
    IFS=$'\t' read -r job_idx tag log_path cmd <<< "${line}"
    status_file="${DONE_DIR}/${job_idx}_${tag}.status"
    if [[ -f "${status_file}" ]]; then
      continue
    fi
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

cat > "${LAUNCH_ROOT}/sbatch_array.sh" <<EOF
#!/bin/bash
#SBATCH -J corr_res_c100_r18
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --output=${LAUNCH_ROOT}/slurm/%A_%a.out
#SBATCH --error=${LAUNCH_ROOT}/slurm/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

bash "${LAUNCH_ROOT}/worker.sh" "${COMMANDS_TSV}" "${LAUNCH_ROOT}/state" "${PROCS_PER_GPU}" "${CONDA_ENV}"
EOF
chmod +x "${LAUNCH_ROOT}/sbatch_array.sh"

cat > "${LAUNCH_ROOT}/submit.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -rf "${LAUNCH_ROOT}/state"
mkdir -p "${LAUNCH_ROOT}/state"
sbatch "${LAUNCH_ROOT}/sbatch_array.sh"
EOF
chmod +x "${LAUNCH_ROOT}/submit.sh"

echo "Wrote ${job_idx} commands:"
echo "  ${COMMANDS_TSV}"
echo "Submit with:"
echo "  bash ${LAUNCH_ROOT}/submit.sh"
echo
echo "One-shot command:"
echo "  PROCS_PER_GPU=${PROCS_PER_GPU} NUM_WORKERS=${NUM_WORKERS} CONDA_ENV=${CONDA_ENV} bash ${ROOT_DIR}/scripts/run_corr_residual_cifar100_resnet18_sbatch.sh && bash ${LAUNCH_ROOT}/submit.sh"

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--submit" ]]; then
  echo "Usage: bash scripts/run_badge_secant_two_methods_s0_4_sbatch.sh [--dry-run|--submit]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${EXPERIMENTS_ROOT}/badge_secant_base_p75_s0_4_sbatch}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-doyoung_env}"
SEEDS="${SEEDS:-0,1,2,3,4}"
MAX_GPUS="${MAX_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-80G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"

if [[ "${MAX_GPUS}" -gt 8 ]]; then
  echo "MAX_GPUS must be <= 8, got ${MAX_GPUS}" >&2
  exit 2
fi
if [[ "${MAX_GPUS}" -lt 1 ]]; then
  echo "MAX_GPUS must be >= 1, got ${MAX_GPUS}" >&2
  exit 2
fi
if [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "PROCS_PER_GPU must be >= 1, got ${PROCS_PER_GPU}" >&2
  exit 2
fi

mkdir -p "${LAUNCH_ROOT}/logs" "${LAUNCH_ROOT}/slurm"

COMMANDS_TSV="${LAUNCH_ROOT}/commands.tsv"
: > "${COMMANDS_TSV}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

jobs=(
  "cifar100 resnet18 set_ADV_main_resnet18_cifar100 cifar100_resnet18"
  "tinyimagenet resnet18 set_ADV_main_resnet18_tinyimagenet tinyimagenet_resnet18"
  "cifar10 resnet18 set_ADV_main_resnet18_cifar10 cifar10_resnet18"
  "svhn resnet18 set_ADV_main_resnet18_svhn svhn_resnet18"
  "cifar100 small_cnn set_ADV_main_smallcnn_cifar100 cifar100_small_cnn"
  "tinyimagenet small_cnn set_ADV_main_smallcnn_tinyimagenet tinyimagenet_small_cnn"
  "cifar10 small_cnn set_ADV_main_smallcnn_cifar10 cifar10_small_cnn"
  "svhn small_cnn set_ADV_main_smallcnn_svhn svhn_small_cnn"
)

variants=(
  "secant_base ours_badge_secant 0"
  "secant_p75 ours_badge_secant 75"
)

download_arg=()
if [[ "${DOWNLOAD_IF_MISSING}" == "1" || "${DOWNLOAD_IF_MISSING}" == "true" ]]; then
  download_arg=(--download-if-missing)
fi

job_idx=0
for spec in "${jobs[@]}"; do
  read -r dataset model set_dir run_prefix <<< "${spec}"
  for variant in "${variants[@]}"; do
    read -r variant_tag method drop_percent <<< "${variant}"
    if [[ "${drop_percent}" == "0" ]]; then
      out_dir="${EXPERIMENTS_ROOT}/${set_dir}/${run_prefix}_ours_badge_secant_b100_r20"
      prefilter_args="--prefilter-metric none --prefilter-drop-percent 0"
    else
      out_dir="${EXPERIMENTS_ROOT}/${set_dir}/${run_prefix}_ours_badge_secant_p${drop_percent}_b100_r20"
      prefilter_args="--prefilter-metric secant_clean_grad_norm --prefilter-drop-percent ${drop_percent}"
    fi
    log_dir="${out_dir}/logs"
    mkdir -p "${log_dir}"
    for seed in "${SEED_ARRAY[@]}"; do
      run_log="${log_dir}/seed_${seed}.log"
      cmd="${PYTHON_BIN} ${ROOT_DIR}/main.py \
--dataset ${dataset} \
--model ${model} \
--acquisition-method ${method} \
${prefilter_args} \
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
--no-save-checkpoints ${download_arg[*]}"
      printf '%s\t%s\t%s\t%s\n' "${job_idx}" "${dataset}_${model}_${variant_tag}_seed${seed}" "${run_log}" "${cmd}" >> "${COMMANDS_TSV}"
      job_idx=$((job_idx + 1))
    done
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

array_end=$((MAX_GPUS - 1))
cat > "${LAUNCH_ROOT}/sbatch_array.sh" <<EOF
#!/bin/bash
#SBATCH -J badge_sec_s04
#SBATCH --array=0-${array_end}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
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
sbatch \${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_array.sh"
EOF
chmod +x "${LAUNCH_ROOT}/submit.sh"

echo "Wrote ${job_idx} commands:"
echo "  ${COMMANDS_TSV}"
echo "Launch root:"
echo "  ${LAUNCH_ROOT}"
echo "GPU array:"
echo "  0-${array_end} (${MAX_GPUS} GPUs max), PROCS_PER_GPU=${PROCS_PER_GPU}"
echo
echo "Submit with:"
echo "  bash ${LAUNCH_ROOT}/submit.sh"
echo
echo "One-shot command:"
echo "  CONDA_ENV=${CONDA_ENV} MAX_GPUS=${MAX_GPUS} PROCS_PER_GPU=${PROCS_PER_GPU} bash ${ROOT_DIR}/scripts/run_badge_secant_two_methods_s0_4_sbatch.sh --submit"

if [[ "${MODE}" == "--submit" ]]; then
  rm -rf "${LAUNCH_ROOT}/state"
  mkdir -p "${LAUNCH_ROOT}/state"
  sbatch ${SBATCH_EXTRA_ARGS:-} "${LAUNCH_ROOT}/sbatch_array.sh"
fi

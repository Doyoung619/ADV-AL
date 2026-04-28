#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" ]]; then
  echo "Usage: bash scripts/run_server_badge_coreset_baselines.sh [--dry-run|--run]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"
COMMAND_ROOT="${COMMAND_ROOT:-${EXPERIMENTS_ROOT}/server_badge_coreset_commands}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS="${SEEDS:-0,1,2,3,4}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"

mkdir -p "$COMMAND_ROOT"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  GPU_ARRAY=(0)
fi

for gpu in "${GPU_ARRAY[@]}"; do
  : > "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
  chmod +x "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
done
: > "${COMMAND_ROOT}/commands_all.txt"

download_arg=()
if [[ "$DOWNLOAD_IF_MISSING" == "1" || "$DOWNLOAD_IF_MISSING" == "true" ]]; then
  download_arg=(--download-if-missing)
fi

jobs=(
  "cifar10 resnet18 badge experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_badge_b100_r20"
  "cifar10 resnet18 coreset experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_coreset_b100_r20"
  "cifar100 resnet18 badge experiments/set_ADV_main_resnet18_cifar100/cifar100_resnet18_badge_b100_r20"
  "cifar100 resnet18 coreset experiments/set_ADV_main_resnet18_cifar100/cifar100_resnet18_coreset_b100_r20"
  "svhn resnet18 badge experiments/set_ADV_main_resnet18_svhn/svhn_resnet18_badge_b100_r20"
  "svhn resnet18 coreset experiments/set_ADV_main_resnet18_svhn/svhn_resnet18_coreset_b100_r20"
  "tinyimagenet resnet18 badge experiments/set_ADV_main_resnet18_tinyimagenet/tinyimagenet_resnet18_badge_b100_r20"
  "tinyimagenet resnet18 coreset experiments/set_ADV_main_resnet18_tinyimagenet/tinyimagenet_resnet18_coreset_b100_r20"
  "cifar10 small_cnn badge experiments/set_ADV_main_smallcnn_cifar10/cifar10_small_cnn_badge_b100_r20"
  "cifar10 small_cnn coreset experiments/set_ADV_main_smallcnn_cifar10/cifar10_small_cnn_coreset_b100_r20"
  "cifar100 small_cnn badge experiments/set_ADV_main_smallcnn_cifar100/cifar100_small_cnn_badge_p10_b100_r20"
  "cifar100 small_cnn coreset experiments/set_ADV_main_smallcnn_cifar100/cifar100_small_cnn_coreset_p10_b100_r20"
  "svhn small_cnn badge experiments/set_ADV_main_smallcnn_svhn/svhn_small_cnn_badge_b100_r20"
  "svhn small_cnn coreset experiments/set_ADV_main_smallcnn_svhn/svhn_small_cnn_coreset_b100_r20"
  "tinyimagenet small_cnn badge experiments/set_ADV_main_smallcnn_tinyimagenet/tinyimagenet_small_cnn_badge_b100_r20"
  "tinyimagenet small_cnn coreset experiments/set_ADV_main_smallcnn_tinyimagenet/tinyimagenet_small_cnn_coreset_b100_r20"
)

slot_count=$((${#GPU_ARRAY[@]} * JOBS_PER_GPU))
if [[ "$slot_count" -le 0 ]]; then
  slot_count=1
fi

job_idx=0
for spec in "${jobs[@]}"; do
  read -r dataset model method rel_out_dir <<< "$spec"
  out_dir="${ROOT_DIR}/${rel_out_dir}"
  log_dir="${out_dir}/logs"
  mkdir -p "$log_dir"

  for seed in "${SEED_ARRAY[@]}"; do
    gpu_slot=$((job_idx % slot_count))
    gpu="${GPU_ARRAY[$((gpu_slot / JOBS_PER_GPU))]}"
    log_path="${log_dir}/seed_${seed}.log"
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} ${ROOT_DIR}/main.py \
--dataset ${dataset} \
--model ${model} \
--acquisition-method ${method} \
--initial-labeled-size 500 \
--acquisition-size 100 \
--num-rounds 20 \
--epochs-per-round 80 \
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
--no-save-checkpoints ${download_arg[*]} > ${log_path} 2>&1"
    echo "$cmd" >> "${COMMAND_ROOT}/commands_all.txt"
    echo "$cmd" >> "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
    job_idx=$((job_idx + 1))
  done
done

echo "[server-baselines] wrote ${job_idx} commands to ${COMMAND_ROOT}/commands_all.txt"
echo "[server-baselines] latest custom method: ours_secant_logdet_refine"
echo "[server-baselines] p10 variant: ours_secant_logdet_refine_p10"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "[server-baselines] dry run only. Use --run to execute generated GPU scripts."
  exit 0
fi

pids=()
for gpu in "${GPU_ARRAY[@]}"; do
  script="${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
  if [[ -s "$script" ]]; then
    echo "[server-baselines] launching ${script}"
    bash "$script" &
    pids+=("$!")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[server-baselines] all jobs finished"

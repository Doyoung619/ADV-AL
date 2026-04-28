#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--run" ]]; then
  echo "Usage: bash scripts/run_server_ours_secant_prefilter_sweep.sh [--dry-run|--run]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${ROOT_DIR}/experiments}"
COMMAND_ROOT="${COMMAND_ROOT:-${EXPERIMENTS_ROOT}/server_ours_secant_prefilter_sweep_s0_s2}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS="${SEEDS:-0,1,2}"
PREFILTER_DROPS="${PREFILTER_DROPS:-10,25,50}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"

mkdir -p "$COMMAND_ROOT"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a DROP_ARRAY <<< "$PREFILTER_DROPS"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  GPU_ARRAY=(0)
fi
if [[ "$JOBS_PER_GPU" -le 0 ]]; then
  JOBS_PER_GPU=1
fi

: > "${COMMAND_ROOT}/commands_all.txt"

slot_files=()
for gpu in "${GPU_ARRAY[@]}"; do
  : > "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
  chmod +x "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
  for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
    slot_file="${COMMAND_ROOT}/commands_gpu_${gpu}_slot_${slot}.sh"
    : > "$slot_file"
    chmod +x "$slot_file"
    slot_files+=("$slot_file")
  done
done

download_arg=()
if [[ "$DOWNLOAD_IF_MISSING" == "1" || "$DOWNLOAD_IF_MISSING" == "true" ]]; then
  download_arg=(--download-if-missing)
fi

# The order is roughly longest-to-shortest so dynamic or slot-based launchers
# start the expensive ResNet/CIFAR-100 jobs first and keep the tail shorter.
jobs=(
  "cifar100 resnet18 set_ADV_main_resnet18_cifar100 cifar100_resnet18_ours_secant_logdet_refine"
  "tinyimagenet resnet18 set_ADV_main_resnet18_tinyimagenet tinyimagenet_resnet18_ours_secant_logdet_refine"
  "cifar10 resnet18 set_ADV_main_resnet18_cifar10 cifar10_resnet18_ours_secant_logdet_refine"
  "svhn resnet18 set_ADV_main_resnet18_svhn svhn_resnet18_ours_secant_logdet_refine"
  "cifar100 small_cnn set_ADV_main_smallcnn_cifar100 cifar100_small_cnn_ours_secant_logdet_refine"
  "tinyimagenet small_cnn set_ADV_main_smallcnn_tinyimagenet tinyimagenet_small_cnn_ours_secant_logdet_refine"
  "cifar10 small_cnn set_ADV_main_smallcnn_cifar10 cifar10_small_cnn_ours_secant_logdet_refine"
  "svhn small_cnn set_ADV_main_smallcnn_svhn svhn_small_cnn_ours_secant_logdet_refine"
)

job_idx=0
for spec in "${jobs[@]}"; do
  read -r dataset model set_dir run_prefix <<< "$spec"
  for drop in "${DROP_ARRAY[@]}"; do
    out_dir="${EXPERIMENTS_ROOT}/${set_dir}/${run_prefix}_p${drop}_b100_r20"
    log_dir="${out_dir}/logs"
    mkdir -p "$log_dir"
    for seed in "${SEED_ARRAY[@]}"; do
      slot_idx=$((job_idx % ${#slot_files[@]}))
      slot_file="${slot_files[$slot_idx]}"
      slot_base="$(basename "$slot_file")"
      gpu="${slot_base#commands_gpu_}"
      gpu="${gpu%%_slot_*}"
      log_path="${log_dir}/seed_${seed}.log"
      cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} ${ROOT_DIR}/main.py \
--dataset ${dataset} \
--model ${model} \
--acquisition-method ours_secant_logdet_refine \
--prefilter-metric D \
--prefilter-drop-percent ${drop} \
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
--no-save-checkpoints ${download_arg[*]} > ${log_path} 2>&1"
      echo "$cmd" >> "${COMMAND_ROOT}/commands_all.txt"
      echo "$cmd" >> "${COMMAND_ROOT}/commands_gpu_${gpu}.sh"
      echo "$cmd" >> "$slot_file"
      job_idx=$((job_idx + 1))
    done
  done
done

cat > "${COMMAND_ROOT}/run_one_terminal.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
eval "\$(conda shell.bash hook)"
conda activate doyoung_env
export DATA_DIR="\${DATA_DIR:-$DATA_DIR}"
export EXPERIMENTS_ROOT="\${EXPERIMENTS_ROOT:-$EXPERIMENTS_ROOT}"
export GPU_IDS="\${GPU_IDS:-$GPU_IDS}"
export JOBS_PER_GPU="\${JOBS_PER_GPU:-$JOBS_PER_GPU}"
python scripts/launch_experiment_set.py --set-root "$COMMAND_ROOT" --jobs-per-gpu "\$JOBS_PER_GPU" --skip-completed
EOF
chmod +x "${COMMAND_ROOT}/run_one_terminal.sh"

cat > "${COMMAND_ROOT}/sbatch_8gpu.sh" <<EOF
#!/bin/bash
#SBATCH -p
#SBATCH -q
#SBATCH -J adval_ours_secant_p10_p25_p50
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=${COMMAND_ROOT}/slurm_%j.out
#SBATCH --error=${COMMAND_ROOT}/slurm_%j.err

cd "$ROOT_DIR"
eval "\$(conda shell.bash hook)"
conda activate doyoung_env

export DATA_DIR="\${DATA_DIR:-$DATA_DIR}"
export EXPERIMENTS_ROOT="\${EXPERIMENTS_ROOT:-$EXPERIMENTS_ROOT}"
export GPU_IDS="\${GPU_IDS:-$GPU_IDS}"
export JOBS_PER_GPU="\${JOBS_PER_GPU:-$JOBS_PER_GPU}"
export NUM_WORKERS="\${NUM_WORKERS:-$NUM_WORKERS}"

bash scripts/run_server_ours_secant_prefilter_sweep.sh --run
EOF
chmod +x "${COMMAND_ROOT}/sbatch_8gpu.sh"

echo "[ours-secant] wrote ${job_idx} commands to ${COMMAND_ROOT}/commands_all.txt"
echo "[ours-secant] command root: ${COMMAND_ROOT}"
echo "[ours-secant] defaults: GPU_IDS=${GPU_IDS} JOBS_PER_GPU=${JOBS_PER_GPU} SEEDS=${SEEDS} PREFILTER_DROPS=${PREFILTER_DROPS}"
echo "[ours-secant] one-terminal runner: ${COMMAND_ROOT}/run_one_terminal.sh"
echo "[ours-secant] sbatch template: ${COMMAND_ROOT}/sbatch_8gpu.sh"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "[ours-secant] dry run only. Use --run to execute generated slot scripts directly."
  exit 0
fi

pids=()
for slot_file in "${slot_files[@]}"; do
  if [[ -s "$slot_file" ]]; then
    echo "[ours-secant] launching ${slot_file}"
    bash "$slot_file" &
    pids+=("$!")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[ours-secant] all jobs finished"

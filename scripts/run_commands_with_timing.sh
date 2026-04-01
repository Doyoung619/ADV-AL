#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_id> <commands_file> [grep_exclude_pattern] [--no-skip-completed]"
  echo "Example: $0 0 experiments/set_A_main/commands_all.txt \"--model vgg16\""
  exit 1
fi

GPU_ID="$1"
COMMANDS_FILE="$2"
EXCLUDE_PATTERN="${3:-}"
SKIP_COMPLETED=1
if [[ "${4:-}" == "--no-skip-completed" ]]; then
  SKIP_COMPLETED=0
fi

if [[ ! -f "$COMMANDS_FILE" ]]; then
  echo "[runner] commands file not found: $COMMANDS_FILE"
  exit 1
fi

format_hms() {
  local sec="$1"
  local h=$((sec / 3600))
  local m=$(((sec % 3600) / 60))
  local s=$((sec % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

extract_arg_value() {
  local cmd="$1"
  local key="$2"
  local next_is_value=0
  for tok in $cmd; do
    if [[ $next_is_value -eq 1 ]]; then
      echo "$tok"
      return 0
    fi
    if [[ "$tok" == "$key" ]]; then
      next_is_value=1
    fi
  done
  echo "unknown"
}

is_completed_run() {
  local body="$1"
  local output_dir run_name rounds run_dir metrics_path count
  output_dir="$(extract_arg_value "$body" "--output-dir")"
  run_name="$(extract_arg_value "$body" "--run-name")"
  rounds="$(extract_arg_value "$body" "--num_rounds")"
  if [[ "$rounds" == "unknown" ]]; then
    rounds="$(extract_arg_value "$body" "--rounds")"
  fi
  if [[ "$rounds" == "unknown" ]]; then
    rounds="10"
  fi
  if [[ "$output_dir" == "unknown" || "$run_name" == "unknown" ]]; then
    return 1
  fi
  run_dir="${output_dir}/${run_name}"
  metrics_path="${run_dir}/round_metrics.json"
  if [[ ! -f "$metrics_path" ]]; then
    return 1
  fi
  count="$(python3 - <<PY
import json
try:
    with open("$metrics_path", "r", encoding="utf-8") as f:
        rows = json.load(f)
    print(len(rows) if isinstance(rows, list) else -1)
except Exception:
    print(-1)
PY
)"
  if [[ "$count" =~ ^[0-9]+$ ]] && (( count >= rounds + 1 )); then
    return 0
  fi
  return 1
}

run_count=0
skipped_count=0
while IFS= read -r cmd || [[ -n "$cmd" ]]; do
  [[ -z "$cmd" ]] && continue
  if [[ -n "$EXCLUDE_PATTERN" ]] && [[ "$cmd" == *"$EXCLUDE_PATTERN"* ]]; then
    continue
  fi

  body="${cmd#CUDA_VISIBLE_DEVICES=* }"
  seed="$(extract_arg_value "$body" "--seed")"
  run_name="$(extract_arg_value "$body" "--run-name")"
  if [[ $SKIP_COMPLETED -eq 1 ]] && is_completed_run "$body"; then
    skipped_count=$((skipped_count + 1))
    echo "[runner] SKIP  gpu=$GPU_ID seed=$seed run=$run_name reason=completed"
    continue
  fi
  run_count=$((run_count + 1))
  start_ts="$(date +%s)"
  echo "[runner] START #$run_count gpu=$GPU_ID seed=$seed run=$run_name at=$(date '+%F %T')"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" bash -lc "$body"
  rc=$?
  set -e

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  elapsed_hms="$(format_hms "$elapsed")"

  if [[ $rc -ne 0 ]]; then
    echo "[runner] FAIL  #$run_count gpu=$GPU_ID seed=$seed run=$run_name rc=$rc elapsed=$elapsed_hms (${elapsed}s)"
  else
    echo "[runner] DONE  #$run_count gpu=$GPU_ID seed=$seed run=$run_name elapsed=$elapsed_hms (${elapsed}s)"
  fi
done < "$COMMANDS_FILE"

echo "[runner] SUMMARY gpu=$GPU_ID ran=$run_count skipped_completed=$skipped_count"

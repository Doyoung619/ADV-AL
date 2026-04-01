import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.gpu_scheduler import discover_gpu_ids


def _load_commands(path: str) -> List[str]:
    cmds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmds.append(line)
    return cmds


def _strip_cuda_prefix(cmd: str) -> str:
    tokens = cmd.split(maxsplit=1)
    if tokens and tokens[0].startswith("CUDA_VISIBLE_DEVICES=") and len(tokens) > 1:
        return tokens[1]
    return cmd


def _parse_arg(tokens: List[str], key: str, default: str = "") -> str:
    for i, t in enumerate(tokens):
        if t == key and i + 1 < len(tokens):
            return tokens[i + 1]
    return default


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _experiment_tag_from_command(cmd: str) -> str:
    body = _strip_cuda_prefix(cmd)
    try:
        tokens = shlex.split(body)
    except Exception:
        return "unknown"
    output_dir = _parse_arg(tokens, "--output-dir", "")
    run_name = _parse_arg(tokens, "--run-name", "")
    if output_dir and run_name:
        return os.path.join(output_dir, run_name)
    if run_name:
        return run_name
    return "unknown"


def _is_completed_from_command(cmd: str) -> bool:
    body = _strip_cuda_prefix(cmd)
    try:
        tokens = shlex.split(body)
    except Exception:
        return False
    output_dir = _parse_arg(tokens, "--output-dir", "")
    run_name = _parse_arg(tokens, "--run-name", "")
    rounds = int(_parse_arg(tokens, "--num_rounds", _parse_arg(tokens, "--rounds", "10")))
    if output_dir == "" or run_name == "":
        return False
    run_dir = os.path.join(output_dir, run_name)
    path = os.path.join(run_dir, "round_metrics.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return isinstance(rows, list) and len(rows) >= rounds + 1
    except Exception:
        return False


def _pick_gpu(gpu_ids: List[int], running_by_gpu: Dict[int, int], jobs_per_gpu: int) -> int:
    candidates = [g for g in gpu_ids if running_by_gpu.get(g, 0) < jobs_per_gpu]
    if not candidates:
        return -1
    candidates.sort(key=lambda g: running_by_gpu.get(g, 0))
    return int(candidates[0])


def main():
    parser = argparse.ArgumentParser(description="Automatic multi-GPU launcher for one experiment set.")
    parser.add_argument("--set-root", type=str, required=True)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=2.0)
    args = parser.parse_args()

    commands_path = os.path.join(os.path.abspath(args.set_root), "commands_all.txt")
    commands = _load_commands(commands_path)
    if not commands:
        print(f"[launch] no commands found: {commands_path}")
        return

    gpu_ids = discover_gpu_ids()
    if len(gpu_ids) == 0:
        gpu_ids = [0]
    jobs_per_gpu = max(1, int(args.jobs_per_gpu))

    pending = []
    skipped = 0
    for cmd in commands:
        if args.skip_completed and _is_completed_from_command(cmd):
            skipped += 1
            continue
        pending.append(cmd)
    print(f"[launch] set={args.set_root} total={len(commands)} pending={len(pending)} skipped={skipped} gpus={gpu_ids}")

    running = []  # [{"proc": Popen, "gpu": int, "cmd": str, "start": float}]
    running_by_gpu: Dict[int, int] = {g: 0 for g in gpu_ids}
    failures = 0

    while pending or running:
        # Spawn new jobs while slots exist.
        while pending:
            gid = _pick_gpu(gpu_ids=gpu_ids, running_by_gpu=running_by_gpu, jobs_per_gpu=jobs_per_gpu)
            if gid < 0:
                break
            cmd = pending.pop(0)
            body = _strip_cuda_prefix(cmd)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gid)
            proc = subprocess.Popen(body, shell=True, env=env)
            running.append({"proc": proc, "gpu": gid, "cmd": body, "start": time.time()})
            running_by_gpu[gid] = running_by_gpu.get(gid, 0) + 1
            exp_tag = _experiment_tag_from_command(cmd)
            print(f"[launch] START gpu={gid} pid={proc.pid} run={exp_tag} cmd={body[:140]}...")

        # Poll running jobs.
        still_running = []
        for item in running:
            proc = item["proc"]
            rc = proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            gid = int(item["gpu"])
            running_by_gpu[gid] = max(0, running_by_gpu.get(gid, 0) - 1)
            elapsed = time.time() - float(item["start"])
            exp_tag = _experiment_tag_from_command(item["cmd"])
            elapsed_hms = _format_elapsed(elapsed)
            if rc != 0:
                failures += 1
                print(f"[launch] FAIL gpu={gid} pid={proc.pid} rc={rc} run={exp_tag} elapsed={elapsed_hms} ({elapsed:.1f}s)")
            else:
                print(f"[launch] DONE gpu={gid} pid={proc.pid} run={exp_tag} elapsed={elapsed_hms} ({elapsed:.1f}s)")
        running = still_running
        time.sleep(float(args.poll_sec))

    if failures > 0:
        raise SystemExit(f"[launch] completed with {failures} failures.")
    print("[launch] completed successfully.")


if __name__ == "__main__":
    main()

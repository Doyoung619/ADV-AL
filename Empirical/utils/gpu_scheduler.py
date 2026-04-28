import os
import subprocess
from collections import defaultdict
from typing import Dict, List


def discover_gpu_ids() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        ids = []
        for line in out.strip().splitlines():
            line = line.strip()
            if line == "":
                continue
            ids.append(int(line))
        if ids:
            return ids
    except Exception:
        pass

    try:
        import torch

        n = torch.cuda.device_count()
        if n > 0:
            return list(range(n))
    except Exception:
        pass
    return []


def partition_commands_by_gpu(commands: List[str], gpu_ids: List[int], jobs_per_gpu: int = 1) -> Dict[int, List[str]]:
    if jobs_per_gpu <= 0:
        raise ValueError(f"jobs_per_gpu must be positive, got {jobs_per_gpu}")
    if len(gpu_ids) == 0:
        return {0: commands.copy()}

    slots: List[int] = []
    for gid in gpu_ids:
        for _ in range(jobs_per_gpu):
            slots.append(gid)
    if len(slots) == 0:
        slots = gpu_ids.copy()

    out: Dict[int, List[str]] = defaultdict(list)
    for i, cmd in enumerate(commands):
        gid = slots[i % len(slots)]
        out[gid].append(cmd)
    return dict(out)


def write_gpu_command_scripts(
    commands_by_gpu: Dict[int, List[str]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for gid in sorted(commands_by_gpu.keys()):
        path = os.path.join(out_dir, f"gpu{gid}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            for cmd in commands_by_gpu[gid]:
                f.write(cmd + "\n")

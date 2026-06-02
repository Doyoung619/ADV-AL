import json
import os
import platform
import socket
import subprocess
from typing import Dict, List


def _safe_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _runtime_info() -> Dict[str, str]:
    try:
        import torch

        torch_ver = str(torch.__version__)
        cuda_ver = str(torch.version.cuda)
    except Exception:
        torch_ver = "unknown"
        cuda_ver = "unknown"
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch_ver,
        "cuda_version": cuda_ver,
        "git_commit": _safe_git_commit(),
    }


def _format_metric_table(seed_metrics: Dict[int, Dict[str, float]], metric_keys: List[str]) -> List[str]:
    lines = []
    header = "| seed | " + " | ".join(metric_keys) + " |"
    sep = "|" + "---|" * (len(metric_keys) + 1)
    lines.append(header)
    lines.append(sep)
    for seed in sorted(seed_metrics.keys()):
        row = seed_metrics[seed]
        vals = [f"{float(row.get(k, float('nan'))):.4f}" for k in metric_keys]
        lines.append(f"| {seed} | " + " | ".join(vals) + " |")
    return lines


def write_experiment_summary_md(
    exp_dir: str,
    identity: Dict,
    seed_metrics: Dict[int, Dict[str, float]],
    aggregate: Dict[str, Dict[str, float]],
    missing_seeds: List[int],
    command_example: str,
    clean_curve_path: str,
    robust_curve_path: str,
) -> str:
    os.makedirs(exp_dir, exist_ok=True)
    info = _runtime_info()
    metric_keys = [
        "avg_auc_clean",
        "avg_auc_robust",
        "avg_final_clean",
        "avg_final_robust",
        "avg_best_clean",
        "avg_best_robust",
        "avg_acquisition_scoring_time_sec",
        "avg_round_time_sec",
        "total_time_sec",
    ]

    lines: List[str] = []
    lines.append(f"# Experiment Summary: {identity['experiment_name']}")
    lines.append("")
    lines.append("## Identity")
    lines.append(f"- Set: `{identity['set_name']}`")
    lines.append(f"- Dataset: `{identity['dataset']}`")
    lines.append(f"- Model: `{identity['model']}`")
    lines.append(f"- Method: `{identity['method']}`")
    lines.append(f"- Hyperparameters: `{json.dumps(identity['hyperparameters'], sort_keys=True)}`")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Git commit: `{info['git_commit']}`")
    lines.append(f"- Hostname: `{info['hostname']}`")
    lines.append(f"- Platform: `{info['platform']}`")
    lines.append(f"- PyTorch: `{info['torch_version']}`")
    lines.append(f"- CUDA: `{info['cuda_version']}`")
    lines.append(f"- Example command: `{command_example}`")
    lines.append("")
    lines.append("## Per-seed Results")
    lines.extend(_format_metric_table(seed_metrics, metric_keys))
    lines.append("")
    lines.append("## Aggregated Statistics")
    lines.append("| metric | mean | std | min | max | count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in sorted(aggregate.keys()):
        s = aggregate[key]
        lines.append(
            f"| {key} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {int(s['count'])} |"
        )
    lines.append("")
    if len(seed_metrics) > 0:
        best_seed = max(seed_metrics.keys(), key=lambda k: seed_metrics[k].get("avg_final_robust", float("-inf")))
        lines.append("## Interpretation")
        lines.append(f"- Best seed by `avg_final_robust`: `{best_seed}`")
        lines.append(
            f"- Mean `avg_final_clean`: `{aggregate.get('avg_final_clean', {}).get('mean', float('nan')):.4f}`"
        )
        lines.append(
            f"- Mean `avg_final_robust`: `{aggregate.get('avg_final_robust', {}).get('mean', float('nan')):.4f}`"
        )
        lines.append(
            f"- Mean `avg_auc_clean`: `{aggregate.get('avg_auc_clean', {}).get('mean', float('nan')):.4f}`"
        )
        lines.append(
            f"- Mean `avg_auc_robust`: `{aggregate.get('avg_auc_robust', {}).get('mean', float('nan')):.4f}`"
        )
    if missing_seeds:
        lines.append("")
        lines.append("## Missing Seeds")
        lines.append("- " + ", ".join(str(s) for s in missing_seeds))
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Clean curve: `{clean_curve_path}`")
    lines.append(f"- Robust curve: `{robust_curve_path}`")
    lines.append("- Logs: `logs/`")

    path = os.path.join(exp_dir, "summary.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_set_summary_md(
    set_root: str,
    set_name: str,
    rows: List[Dict],
    missing_by_experiment: Dict[str, List[int]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Set Summary: {set_name}")
    lines.append("")
    lines.append("| experiment | dataset | model | method | avg_final_clean | avg_final_robust | avg_auc_clean | avg_auc_robust |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| [{r['experiment_name']}]({r['experiment_rel_path']}) | {r['dataset']} | {r['model']} | {r['method']} | "
            f"{r.get('avg_final_clean', float('nan')):.4f} | {r.get('avg_final_robust', float('nan')):.4f} | "
            f"{r.get('avg_auc_clean', float('nan')):.4f} | {r.get('avg_auc_robust', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("## Missing Seeds")
    for exp_name, missing in sorted(missing_by_experiment.items()):
        if missing:
            lines.append(f"- `{exp_name}`: {', '.join(str(s) for s in missing)}")
    path = os.path.join(set_root, "summary.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path

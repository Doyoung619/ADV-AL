import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
TARGET_EXPERIMENTS = ["experimentA", "experimentB", "experimentC", "experimentD", "experimentE"]


def _latest_analysis_md_path() -> str:
    latest_path = ""
    latest_mtime = -1.0
    main_root = os.path.join(EXPERIMENTS_ROOT, "experimentMain")
    for dirpath, _, filenames in os.walk(main_root):
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path
    if latest_path == "":
        raise FileNotFoundError("No analysis markdown file found under experimentMain.")
    return latest_path


def _read_latest_analysis_md() -> Tuple[str, str]:
    path = _latest_analysis_md_path()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().rstrip() + "\n"
    return path, content


def _latest_analysis_md_for_container(container_dir: str) -> Tuple[str, str]:
    analysis_root = os.path.join(container_dir, "analysis")
    latest_path = ""
    latest_mtime = -1.0
    if os.path.isdir(analysis_root):
        for dirpath, _, filenames in os.walk(analysis_root):
            for name in filenames:
                if not name.endswith(".md"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path
    if latest_path != "":
        with open(latest_path, "r", encoding="utf-8") as f:
            content = f.read().rstrip() + "\n"
        return latest_path, content
    return _read_latest_analysis_md()


def _dataset_dirs_for_experiment(experiment_root: str) -> List[str]:
    out: List[str] = []
    for setting_name in sorted(os.listdir(experiment_root)):
        setting_path = os.path.join(experiment_root, setting_name)
        if not os.path.isdir(setting_path):
            continue
        for dataset_name in sorted(os.listdir(setting_path)):
            dataset_path = os.path.join(setting_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            out.append(dataset_path)
    return out


def _main_set_dirs(experiment_main_root: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(experiment_main_root)):
        path = os.path.join(experiment_main_root, name)
        if os.path.isdir(path) and name.startswith("set_ADV_main_"):
            out.append(path)
    return out


def _algorithm_dirs(dataset_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if not os.path.isdir(path):
            continue
        if name in {"anothers", "_graphs"}:
            continue
        if any(entry.startswith("seed_") and os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path)):
            out.append(path)
    return out


def _read_round_rows_from_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return []
    rows = [row for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("round", 0)))
    return rows


def _read_round_rows_from_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row.get("round", 0)))
    return rows


def _read_seed_rows(seed_dir: str) -> List[Dict]:
    json_path = os.path.join(seed_dir, "round_metrics.json")
    csv_path = os.path.join(seed_dir, "round_metrics.csv")
    if os.path.exists(json_path):
        return _read_round_rows_from_json(json_path)
    if os.path.exists(csv_path):
        return _read_round_rows_from_csv(csv_path)
    return []


def _to_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _robust_from_row(row: Dict) -> float:
    for key in ("pgd_acc", "pgd10_acc", "pgd_robust_acc", "fgsm_acc"):
        if key in row:
            return _to_float(row.get(key))
    return float("nan")


def _pgd_from_row(row: Dict) -> float:
    for key in ("pgd_acc", "pgd10_acc", "pgd_robust_acc"):
        if key in row:
            return _to_float(row.get(key))
    return float("nan")


def _algorithm_label_from_dirname(name: str) -> str:
    lowered = name.lower()
    if "ours_badge_secant_p75" in lowered:
        return "ours + p75"
    if "ours_badge_secant" in lowered:
        return "ours"
    if "batchbald" in lowered:
        return "BatchBALD"
    if "entropy" in lowered:
        return "Entropy"
    if "random" in lowered:
        return "Random"
    if "saal" in lowered:
        return "SAAL"
    if "badge_p10" in lowered:
        return "BADGE p10"
    if "badge" in lowered:
        return "BADGE"
    return name


def _dataset_display_name(raw: str) -> str:
    lowered = raw.lower()
    mapping = {
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
        "tinyimagenet": "TinyImageNet",
        "svhn": "SVHN",
    }
    return mapping.get(lowered, raw)


def _model_display_name(raw: str) -> str:
    lowered = raw.lower().replace("-", "").replace("_", "")
    if "resnet18" in lowered or lowered == "resnet":
        return "ResNet"
    if "smallcnn" in lowered:
        return "CNN"
    if "vgg16" in lowered:
        return "VGG"
    return raw


def _infer_title_parts(container_dir: str, algorithm_dirs: Sequence[str]) -> Tuple[str, str, str]:
    base = os.path.basename(container_dir)
    parent = os.path.basename(os.path.dirname(container_dir))

    dataset_raw = ""
    model_raw = ""
    batch_size = ""

    main_match = re.match(r"set_ADV_main_([a-z0-9]+)_([a-z0-9]+)$", base)
    if main_match is not None:
        model_raw = main_match.group(1)
        dataset_raw = main_match.group(2)
    else:
        parts = base.split("_")
        if len(parts) >= 3:
            dataset_raw = parts[-1]
            model_raw = "_".join(parts[:-1])
        else:
            dataset_raw = base
            model_raw = base

    acq_match = re.search(r"acq(\d+)", parent)
    if acq_match is not None:
        batch_size = acq_match.group(1)
    else:
        for algorithm_dir in algorithm_dirs:
            algo_name = os.path.basename(algorithm_dir)
            batch_match = re.search(r"_b(\d+)_r\d+$", algo_name)
            if batch_match is not None:
                batch_size = batch_match.group(1)
                break

    return _dataset_display_name(dataset_raw), _model_display_name(model_raw), batch_size or "unknown"


def _collect_metric_curves(algorithm_dir: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    clean_curves: List[np.ndarray] = []
    pgd_curves: List[np.ndarray] = []
    seeds: List[int] = []

    for name in sorted(os.listdir(algorithm_dir)):
        if not name.startswith("seed_"):
            continue
        seed_dir = os.path.join(algorithm_dir, name)
        if not os.path.isdir(seed_dir):
            continue
        rows = _read_seed_rows(seed_dir)
        if not rows:
            continue
        clean_curve = np.asarray([_to_float(row.get("clean_acc")) for row in rows], dtype=np.float64)
        pgd_curve = np.asarray([_pgd_from_row(row) for row in rows], dtype=np.float64)
        clean_curves.append(clean_curve)
        pgd_curves.append(pgd_curve)
        try:
            seeds.append(int(name.split("_", 1)[1]))
        except Exception:
            pass

    if not clean_curves:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64), []

    min_len = min(curve.shape[0] for curve in clean_curves)
    clean = np.stack([curve[:min_len] for curve in clean_curves], axis=0)
    pgd = np.stack([curve[:min_len] for curve in pgd_curves], axis=0)
    return clean, pgd, seeds


def _summarize_curves(curves: np.ndarray) -> List[Dict]:
    if curves.size == 0:
        return []
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    rows: List[Dict] = []
    for round_idx in range(mean.shape[0]):
        ci_band = float(std[round_idx] / 2.0)
        rows.append(
            {
                "round": round_idx,
                "mean": float(mean[round_idx]),
                "std": float(std[round_idx]),
                "ci_band": ci_band,
                "lower_ci": float(mean[round_idx] - ci_band),
                "upper_ci": float(mean[round_idx] + ci_band),
                "seed_count": int(curves.shape[0]),
            }
        )
    return rows


def _write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric(
    out_path: str,
    title: str,
    ylabel: str,
    algorithm_rows: List[Tuple[str, List[Dict]]],
) -> None:
    plt.figure(figsize=(8.2, 5.5))
    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]
    preferred_slots = {
        "ours": 0,
        "ours + p75": 3,
    }
    used_slots = set()
    color_by_label: Dict[str, Tuple[float, float, float, float]] = {}
    for label, _ in algorithm_rows:
        if label in preferred_slots:
            slot = preferred_slots[label]
            color_by_label[label] = base_colors[slot]
            used_slots.add(slot)
    next_slots = [idx for idx in range(len(base_colors)) if idx not in used_slots]
    next_slot_idx = 0
    for label, _ in algorithm_rows:
        if label in color_by_label:
            continue
        if next_slot_idx >= len(next_slots):
            slot = len(color_by_label) % len(base_colors)
        else:
            slot = next_slots[next_slot_idx]
            next_slot_idx += 1
        color_by_label[label] = base_colors[slot]

    plotted = False
    for idx, (label, rows) in enumerate(algorithm_rows):
        if not rows:
            continue
        x = np.asarray([int(row["round"]) for row in rows], dtype=np.int64)
        y = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
        y_ci = np.asarray([float(row["ci_band"]) for row in rows], dtype=np.float64)
        color = color_by_label[label]
        plt.plot(x, y, label=label, color=color, linewidth=2)
        plt.fill_between(x, y - y_ci, y + y_ci, color=color, alpha=0.18)
        if int(x.min()) == int(x.max()):
            plt.xlim(float(x.min()) - 0.5, float(x.max()) + 0.5)
        else:
            plt.xlim(int(x.min()), int(x.max()))
        plotted = True
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    plt.xlabel("Round", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(alpha=0.3)
    plt.margins(x=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if plotted:
        plt.legend(fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def _write_graphs_md(
    graphs_dir: str,
    dataset_dir: str,
    algorithms: List[str],
    clean_csv_name: str,
    robust_csv_name: str,
    latest_analysis_path: str,
    latest_analysis_md: str,
) -> None:
    path = os.path.join(graphs_dir, "README.md")
    lines: List[str] = []
    lines.append(f"# Graph Bundle: {os.path.basename(dataset_dir)}")
    lines.append("")
    lines.append(f"- Source dataset dir: `{dataset_dir}`")
    lines.append(f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip()}`")
    lines.append(f"- Included algorithms: `{', '.join(algorithms)}`")
    lines.append("- Excluded subtree: `anothers/`")
    lines.append("- PGD metric priority: `pgd_acc`, fallback to `pgd10_acc`, then `pgd_robust_acc`")
    lines.append("")
    lines.append("## Files")
    lines.append("- `clean_acc_rounds.png`")
    lines.append("- `pgd_acc_rounds.png`")
    lines.append(f"- `{clean_csv_name}`")
    lines.append(f"- `{robust_csv_name}`")
    lines.append("")
    lines.append("## Latest Analysis Source")
    lines.append(f"- `{latest_analysis_path}`")
    lines.append("")
    lines.append("## Copied Analysis Markdown")
    lines.append("")
    lines.append(latest_analysis_md.rstrip())
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def build_graphs_for_container_dir(
    container_dir: str,
    latest_analysis_path: str,
    latest_analysis_md: str,
) -> Optional[str]:
    algorithm_dirs = _algorithm_dirs(container_dir)
    if not algorithm_dirs:
        return None

    graphs_dir = os.path.join(container_dir, "_graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    for stale_name in ("robust_acc_rounds.png", "robust_acc_round_stats.csv"):
        stale_path = os.path.join(graphs_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    clean_csv_rows: List[Dict] = []
    pgd_csv_rows: List[Dict] = []
    clean_plot_rows: List[Tuple[str, List[Dict]]] = []
    pgd_plot_rows: List[Tuple[str, List[Dict]]] = []
    algorithm_names: List[str] = []
    dataset_display, model_display, batch_size = _infer_title_parts(container_dir, algorithm_dirs)

    for algorithm_dir in algorithm_dirs:
        algorithm_name = _algorithm_label_from_dirname(os.path.basename(algorithm_dir))
        clean_curves, pgd_curves, seeds = _collect_metric_curves(algorithm_dir)
        if clean_curves.size == 0:
            continue
        clean_rows = _summarize_curves(clean_curves)
        pgd_rows = _summarize_curves(pgd_curves)
        for row in clean_rows:
            clean_csv_rows.append({"algorithm": algorithm_name, **row})
        for row in pgd_rows:
            pgd_csv_rows.append({"algorithm": algorithm_name, **row})
        clean_plot_rows.append((algorithm_name, clean_rows))
        pgd_plot_rows.append((algorithm_name, pgd_rows))
        algorithm_names.append(algorithm_name)

    if not algorithm_names:
        return None

    clean_csv_name = "clean_acc_round_stats.csv"
    robust_csv_name = "pgd_acc_round_stats.csv"
    _write_csv(
        os.path.join(graphs_dir, clean_csv_name),
        clean_csv_rows,
        ["algorithm", "round", "mean", "std", "ci_band", "lower_ci", "upper_ci", "seed_count"],
    )
    _write_csv(
        os.path.join(graphs_dir, robust_csv_name),
        pgd_csv_rows,
        ["algorithm", "round", "mean", "std", "ci_band", "lower_ci", "upper_ci", "seed_count"],
    )

    _plot_metric(
        os.path.join(graphs_dir, "clean_acc_rounds.png"),
        f"{dataset_display}, {model_display}, Batch size : {batch_size}",
        "Clean Accuracy (%)",
        clean_plot_rows,
    )
    _plot_metric(
        os.path.join(graphs_dir, "pgd_acc_rounds.png"),
        f"{dataset_display}, {model_display}, Batch size : {batch_size}",
        "PGD Accuracy (%)",
        pgd_plot_rows,
    )
    _write_graphs_md(
        graphs_dir=graphs_dir,
        dataset_dir=container_dir,
        algorithms=algorithm_names,
        clean_csv_name=clean_csv_name,
        robust_csv_name=robust_csv_name,
        latest_analysis_path=latest_analysis_path,
        latest_analysis_md=latest_analysis_md,
    )
    return graphs_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build _graphs bundles under experimentA-E dataset directories.")
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(TARGET_EXPERIMENTS),
        help="Comma-separated experiment roots under experiments/.",
    )
    parser.add_argument(
        "--include-main",
        action="store_true",
        help="Also build _graphs bundles for experimentMain set roots.",
    )
    args = parser.parse_args()

    target_names = [name.strip() for name in args.experiments.split(",") if name.strip()]

    generated: List[str] = []
    for experiment_name in target_names:
        experiment_root = os.path.join(EXPERIMENTS_ROOT, experiment_name)
        if not os.path.isdir(experiment_root):
            continue
        for dataset_dir in _dataset_dirs_for_experiment(experiment_root):
            latest_analysis_path, latest_analysis_md = _latest_analysis_md_for_container(dataset_dir)
            out_dir = build_graphs_for_container_dir(
                container_dir=dataset_dir,
                latest_analysis_path=latest_analysis_path,
                latest_analysis_md=latest_analysis_md,
            )
            if out_dir is not None:
                generated.append(out_dir)

    if args.include_main:
        experiment_main_root = os.path.join(EXPERIMENTS_ROOT, "experimentMain")
        if os.path.isdir(experiment_main_root):
            for set_dir in _main_set_dirs(experiment_main_root):
                latest_analysis_path, latest_analysis_md = _latest_analysis_md_for_container(set_dir)
                out_dir = build_graphs_for_container_dir(
                    container_dir=set_dir,
                    latest_analysis_path=latest_analysis_path,
                    latest_analysis_md=latest_analysis_md,
                )
                if out_dir is not None:
                    generated.append(out_dir)

    print(f"[graphs] generated {len(generated)} _graphs directories")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

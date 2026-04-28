import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT_ROOT = os.path.join(
    PROJECT_ROOT, "experiments", "experimentF", "init500_acq100_round20"
)
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "experiments", "figures", "experimentF")
PERCENTILES = (10, 25, 50, 75, 90)
GROUPS = ("base", *(f"p{p}" for p in PERCENTILES))
GROUP_DIRS = ("ours_badge_secant", *(f"ours_badge_secant_p{p}" for p in PERCENTILES))


@dataclass
class SeedMetrics:
    clean_acc: float
    clean_auc: float
    robust_acc: float
    robust_auc: float


def _to_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_round_metrics(seed_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(seed_dir, "round_metrics.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(float(row.get("round", 0))))
    return rows


def _series_auc(rows: Sequence[Dict[str, str]], key: str) -> float:
    pairs = []
    for row in rows:
        x = _to_float(row.get("round"))
        y = _to_float(row.get(key))
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    span = float(x[-1] - x[0])
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(y, x) / span)


def _final_value(rows: Sequence[Dict[str, str]], key: str) -> float:
    for row in reversed(rows):
        value = _to_float(row.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _seed_metrics(seed_dir: str) -> SeedMetrics:
    rows = _read_round_metrics(seed_dir)
    return SeedMetrics(
        clean_acc=_final_value(rows, "clean_acc"),
        clean_auc=_series_auc(rows, "clean_acc"),
        robust_acc=_final_value(rows, "pgd_acc"),
        robust_auc=_series_auc(rows, "pgd_acc"),
    )


def _seed_dirs(method_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(method_dir)):
        path = os.path.join(method_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"seed_\d+", name):
            out.append(path)
    return out


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _display_model_dataset(name: str) -> tuple[str, str]:
    model, dataset = name.rsplit("_", 1)
    model_map = {"small_cnn": "CNN", "resnet18": "ResNet18"}
    dataset_map = {"CIFAR10": "CIFAR10", "CIFAR100": "CIFAR100"}
    return dataset_map.get(dataset, dataset), model_map.get(model, model)


def _tight_ylim(means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    low = float(np.nanmin(means - stds))
    high = float(np.nanmax(means + stds))
    if not math.isfinite(low) or not math.isfinite(high):
        return 0.0, 100.0

    padding = max(0.5, (high - low) * 0.06)
    ymin = max(0.0, math.floor((low - padding)))
    ymax = min(100.0, math.ceil((high + padding)))
    return ymin, ymax


def _collect_means_stds(container_dir: str, attrs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(GROUPS), len(attrs)), np.nan)
    stds = np.full_like(means, np.nan)
    for g_idx, group_dir in enumerate(GROUP_DIRS):
        method_dir = os.path.join(container_dir, group_dir)
        metrics = [_seed_metrics(seed_dir) for seed_dir in _seed_dirs(method_dir)]
        for m_idx, attr in enumerate(attrs):
            means[g_idx, m_idx], stds[g_idx, m_idx] = _mean_std(
                [getattr(metric, attr) for metric in metrics]
            )
    return means, stds


def _plot_pair(
    container_dir: str,
    output_dir: str,
    left_attr: str,
    right_attr: str,
    left_label: str,
    right_label: str,
    suffix: str,
) -> str:
    means, stds = _collect_means_stds(container_dir, (left_attr, right_attr))

    x = np.arange(len(GROUPS), dtype=float) * 0.82
    width = 0.32
    gap = width * 0.55

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax_left = plt.subplots(figsize=(8.4, 6.4))
    ax_right = ax_left.twinx()

    color_left = "#1f77b4"
    color_right = "#d62728"

    ax_left.bar(
        x - gap, means[:, 0], width,
        label=left_label,
        color=color_left, edgecolor="black", linewidth=0.6,
    )
    ax_right.bar(
        x + gap, means[:, 1], width,
        label=right_label,
        color=color_right, edgecolor="black", linewidth=0.6,
    )

    base_left = means[0, 0]
    base_right = means[0, 1]
    if math.isfinite(base_left):
        ax_left.axhline(
            base_left, color=color_left, linestyle="--", linewidth=1.3, alpha=0.7,
            label=f"{left_label} (base)",
        )
    if math.isfinite(base_right):
        ax_right.axhline(
            base_right, color=color_right, linestyle="--", linewidth=1.3, alpha=0.7,
            label=f"{right_label} (base)",
        )

    dataset, model = _display_model_dataset(os.path.basename(container_dir))
    ax_left.set_title(f"{dataset}, {model}, Batch size : 100", pad=12)
    ax_left.set_xlabel("Prefilter percentile")
    ax_left.set_ylabel(f"{left_label} (%)", color=color_left)
    ax_right.set_ylabel(f"{right_label} (%)", color=color_right)
    ax_left.tick_params(axis="y", labelcolor=color_left)
    ax_right.tick_params(axis="y", labelcolor=color_right)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(list(GROUPS))

    left_lim = _tight_ylim(means[:, 0:1], stds[:, 0:1])
    right_lim = _tight_ylim(means[:, 1:2], stds[:, 1:2])
    left_span = left_lim[1] - left_lim[0]
    right_span = right_lim[1] - right_lim[0]
    base_span = max(left_span, right_span)
    left_target = base_span
    right_target = base_span * 1.6
    left_center = (left_lim[0] + left_lim[1]) / 2.0
    right_center = (right_lim[0] + right_lim[1]) / 2.0
    ax_left.set_ylim(max(0.0, left_center - left_target / 2.0), min(100.0, left_center + left_target / 2.0))
    ax_right.set_ylim(max(0.0, right_center - right_target / 2.0), min(100.0, right_center + right_target / 2.0))

    ax_left.grid(axis="y", alpha=0.25, linewidth=0.8)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        ncol=2, loc="upper left", frameon=True, framealpha=0.9,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"init500_acq100_round20__{os.path.basename(container_dir)}__{suffix}.pdf",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_container(container_dir: str, output_dir: str) -> List[str]:
    return [
        _plot_pair(
            container_dir, output_dir,
            left_attr="clean_acc", right_attr="clean_auc",
            left_label="Clean Acc", right_label="Clean AUC",
            suffix="clean_bars",
        ),
        _plot_pair(
            container_dir, output_dir,
            left_attr="robust_acc", right_attr="robust_auc",
            left_label="PGD Acc", right_label="PGD AUC",
            suffix="pgd_bars",
        ),
    ]


def build_figures(input_root: str, output_root: str) -> List[str]:
    outputs = []
    for name in sorted(os.listdir(input_root)):
        container_dir = os.path.join(input_root, name)
        if not os.path.isdir(container_dir) or "_" not in name:
            continue
        if any(
            os.path.isdir(os.path.join(container_dir, group_dir))
            for group_dir in GROUP_DIRS
        ):
            outputs.extend(_plot_container(container_dir, output_root))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ExperimentF p10-p90 bar figures.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    outputs = build_figures(args.input_root, args.output_root)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

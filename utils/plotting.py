import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_curves(clean_curves: np.ndarray, robust_curves: np.ndarray, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, "clean_curve.png")
    robust_path = os.path.join(out_dir, "robust_curve.png")

    if clean_curves.size > 0:
        mean_clean = np.nanmean(clean_curves, axis=0)
        std_clean = np.nanstd(clean_curves, axis=0)
        x = np.arange(mean_clean.size)
        plt.figure(figsize=(7, 4))
        plt.plot(x, mean_clean, label="mean clean")
        plt.fill_between(x, mean_clean - std_clean, mean_clean + std_clean, alpha=0.2)
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title("Clean Accuracy Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(clean_path, dpi=200)
        plt.close()
    else:
        clean_path = ""

    if robust_curves.size > 0:
        mean_robust = np.nanmean(robust_curves, axis=0)
        std_robust = np.nanstd(robust_curves, axis=0)
        x = np.arange(mean_robust.size)
        plt.figure(figsize=(7, 4))
        plt.plot(x, mean_robust, label="mean robust")
        plt.fill_between(x, mean_robust - std_robust, mean_robust + std_robust, alpha=0.2)
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title("Robust Accuracy Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(robust_path, dpi=200)
        plt.close()
    else:
        robust_path = ""

    return {"clean_curve": clean_path, "robust_curve": robust_path}


def plot_set_group_curves(set_root: str, grouped: Dict[str, List[Dict]], metric_key: str, out_name: str, title: str) -> str:
    out_path = os.path.join(set_root, out_name)
    plt.figure(figsize=(9, 5))
    plotted = False
    for group_name in sorted(grouped.keys()):
        rows = grouped[group_name]
        for row in rows:
            curve = row.get(metric_key, [])
            if not curve:
                continue
            x = np.arange(len(curve))
            label = f"{group_name}:{row['method']}"
            plt.plot(x, curve, label=label)
            plotted = True
    if plotted:
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_set_bar(
    set_root: str,
    rows: List[Dict],
    metric: str,
    out_name: str,
    title: str,
) -> str:
    out_path = os.path.join(set_root, out_name)
    names = [r["experiment_name"] for r in rows]
    vals = [float(r.get(metric, float("nan"))) for r in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(max(10, len(names) * 0.25), 5))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=90)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_percentile_sweep(set_root: str, rows: List[Dict], metric: str, out_name: str, title: str) -> str:
    out_path = os.path.join(set_root, out_name)
    plt.figure(figsize=(8, 5))
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        key = f"{r['dataset']}_{r['method'].split('_')[1]}"
        groups.setdefault(key, []).append(r)

    for key in sorted(groups.keys()):
        rr = groups[key]
        rr_sorted = sorted(rr, key=lambda x: int(x.get("percentile", 0)))
        x = [int(r.get("percentile", 0)) for r in rr_sorted]
        y = [float(r.get(metric, float("nan"))) for r in rr_sorted]
        plt.plot(x, y, marker="o", label=key)

    plt.xlabel("Percentile")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

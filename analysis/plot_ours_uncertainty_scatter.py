import argparse
import csv
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.zeros(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return float("nan")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _load_analysis_table(path: str) -> Dict[str, np.ndarray]:
    cols = {
        "our_score": [],
        "uncertainty_score": [],
        "selected_pure": [],
        "selected_filtered": [],
    }
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["our_score"].append(float(row["our_score"]))
            cols["uncertainty_score"].append(float(row["uncertainty_score"]))
            cols["selected_pure"].append(int(row["selected_pure"]))
            cols["selected_filtered"].append(int(row["selected_filtered"]))
    return {k: np.asarray(v) for k, v in cols.items()}


def main():
    p = argparse.ArgumentParser(description="Plot our score vs uncertainty from analysis_table.csv")
    p.add_argument("--run-dir", type=str, required=True, help="Path to output_Experiment2/<run_dir>")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    table_path = os.path.join(run_dir, "analysis_table.csv")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"analysis_table.csv not found: {table_path}")

    data = _load_analysis_table(table_path)
    ours = data["our_score"]
    unc = data["uncertainty_score"]
    pure = data["selected_pure"] == 1
    filt = data["selected_filtered"] == 1

    pearson = float(np.corrcoef(ours, unc)[0, 1]) if len(ours) > 1 else float("nan")
    spearman = _spearman_corr(ours, unc)

    fig_path = os.path.join(run_dir, "figure_c_ours_vs_uncertainty.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(ours, unc, s=10, c="gray", alpha=0.25, label="all candidates")
    plt.scatter(ours[pure], unc[pure], s=18, c="red", alpha=0.8, label="pure ours")
    plt.scatter(ours[filt], unc[filt], s=18, c="blue", alpha=0.8, label="filtered ours")
    plt.xlabel("Our Score")
    plt.ylabel("Uncertainty Score")
    plt.title("Our score vs. uncertainty")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    corr_path = os.path.join(run_dir, "ours_uncertainty_correlation.json")
    with open(corr_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "n_samples": int(len(ours)),
                "pearson_our_vs_uncertainty": pearson,
                "spearman_our_vs_uncertainty": spearman,
                "figure_path": fig_path,
            },
            f,
            indent=2,
        )

    print(f"Saved: {fig_path}")
    print(f"Saved: {corr_path}")
    print(f"Pearson: {pearson:.6f}")
    print(f"Spearman: {spearman:.6f}")


if __name__ == "__main__":
    main()


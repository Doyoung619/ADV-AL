import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.markdown_summary import write_experiment_summary_md, write_set_summary_md
from utils.metrics_aggregation import summarize_experiment_dir
from utils.plotting import plot_experiment_curves, plot_percentile_sweep, plot_set_bar, plot_set_group_curves


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_experiment_dirs(set_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(set_root)):
        path = os.path.join(set_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "config.json")):
            out.append(path)
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate one experiment set.")
    parser.add_argument("--set-root", type=str, required=True)
    args = parser.parse_args()

    set_root = os.path.abspath(args.set_root)
    set_name = os.path.basename(set_root.rstrip("\\/"))
    exp_dirs = _discover_experiment_dirs(set_root)
    if not exp_dirs:
        print(f"[aggregate] no experiments found under {set_root}")
        return

    comparison_rows = []
    missing_by_experiment: Dict[str, List[int]] = {}
    grouped_for_curves: Dict[str, List[Dict]] = {}
    percentile_rows: List[Dict] = []

    for exp_dir in exp_dirs:
        cfg = _load_json(os.path.join(exp_dir, "config.json"))
        seeds = [int(s) for s in cfg["seeds"]]
        summary = summarize_experiment_dir(exp_dir=exp_dir, seed_list=seeds)
        seed_metrics = summary["seed_metrics"]
        aggregate = summary["aggregate"]
        missing = summary["missing_seeds"]
        clean_curves = summary["clean_curves"]
        robust_curves = summary["robust_curves"]

        curve_paths = plot_experiment_curves(clean_curves=clean_curves, robust_curves=robust_curves, out_dir=exp_dir)
        clean_mean = np.nanmean(clean_curves, axis=0).tolist() if clean_curves.size > 0 else []
        robust_mean = np.nanmean(robust_curves, axis=0).tolist() if robust_curves.size > 0 else []

        per_seed_rows = []
        for seed in sorted(seed_metrics.keys()):
            row = {"seed": int(seed)}
            row.update(seed_metrics[seed])
            per_seed_rows.append(row)
        _write_csv(os.path.join(exp_dir, "seed_metrics.csv"), per_seed_rows)

        aggregate_json = {
            "identity": cfg,
            "missing_seeds": missing,
            "seed_metrics": seed_metrics,
            "aggregate": aggregate,
            "curve_mean_clean": clean_mean,
            "curve_mean_robust": robust_mean,
        }
        with open(os.path.join(exp_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(aggregate_json, f, indent=2)

        aggregate_rows = []
        for metric, stats in sorted(aggregate.items()):
            aggregate_rows.append({"metric": metric, **stats})
        _write_csv(os.path.join(exp_dir, "aggregate_metrics.csv"), aggregate_rows)

        commands_path = os.path.join(exp_dir, "commands.txt")
        command_example = ""
        if os.path.exists(commands_path):
            with open(commands_path, "r", encoding="utf-8") as f:
                command_example = (f.readline() or "").strip()

        write_experiment_summary_md(
            exp_dir=exp_dir,
            identity={
                "experiment_name": cfg["experiment_name"],
                "set_name": cfg["set_name"],
                "dataset": cfg["dataset"],
                "model": cfg["model"],
                "method": cfg["method"],
                "hyperparameters": {
                    "initial_labeled_size": cfg["initial_labeled_size"],
                    "acquisition_size": cfg["acquisition_size"],
                    "rounds": cfg["rounds"],
                    "epochs_per_round": cfg["epochs_per_round"],
                    "eval_epsilon": cfg["eval_epsilon"],
                },
            },
            seed_metrics=seed_metrics,
            aggregate=aggregate,
            missing_seeds=missing,
            command_example=command_example,
            clean_curve_path=os.path.relpath(curve_paths["clean_curve"], exp_dir) if curve_paths["clean_curve"] else "",
            robust_curve_path=os.path.relpath(curve_paths["robust_curve"], exp_dir) if curve_paths["robust_curve"] else "",
        )

        row = {
            "experiment_name": cfg["experiment_name"],
            "experiment_rel_path": os.path.relpath(exp_dir, set_root).replace("\\", "/"),
            "dataset": cfg["dataset"],
            "model": cfg["model"],
            "method": cfg["method"],
            "avg_auc_clean": aggregate.get("avg_auc_clean", {}).get("mean", float("nan")),
            "avg_auc_robust": aggregate.get("avg_auc_robust", {}).get("mean", float("nan")),
            "avg_final_clean": aggregate.get("avg_final_clean", {}).get("mean", float("nan")),
            "avg_final_robust": aggregate.get("avg_final_robust", {}).get("mean", float("nan")),
            "avg_best_clean": aggregate.get("avg_best_clean", {}).get("mean", float("nan")),
            "avg_best_robust": aggregate.get("avg_best_robust", {}).get("mean", float("nan")),
            "missing_seed_count": len(missing),
        }
        comparison_rows.append(row)
        missing_by_experiment[cfg["experiment_name"]] = missing

        group_key = f"{cfg['dataset']}_{cfg['model']}"
        grouped_for_curves.setdefault(group_key, []).append(
            {"method": cfg["method"], "curve_mean_clean": clean_mean, "curve_mean_robust": robust_mean}
        )

        if set_name == "set_H_percentile":
            pct = None
            if "_p" in cfg["method"]:
                try:
                    pct = int(cfg["method"].split("_p")[-1])
                except Exception:
                    pct = None
            percentile_rows.append(
                {
                    "dataset": cfg["dataset"],
                    "method": cfg["method"],
                    "percentile": pct if pct is not None else 0,
                    "avg_auc_clean": row["avg_auc_clean"],
                    "avg_auc_robust": row["avg_auc_robust"],
                    "avg_final_clean": row["avg_final_clean"],
                    "avg_final_robust": row["avg_final_robust"],
                }
            )

    _write_csv(os.path.join(set_root, "comparison.csv"), comparison_rows)
    with open(os.path.join(set_root, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    # Set-level plots.
    plot_set_bar(set_root, comparison_rows, "avg_final_clean", "bar_avg_final_clean.png", f"{set_name}: avg_final_clean")
    plot_set_bar(set_root, comparison_rows, "avg_final_robust", "bar_avg_final_robust.png", f"{set_name}: avg_final_robust")
    plot_set_group_curves(
        set_root=set_root,
        grouped=grouped_for_curves,
        metric_key="curve_mean_clean",
        out_name="set_clean_curves.png",
        title=f"{set_name}: clean curves by method",
    )
    plot_set_group_curves(
        set_root=set_root,
        grouped=grouped_for_curves,
        metric_key="curve_mean_robust",
        out_name="set_robust_curves.png",
        title=f"{set_name}: robust curves by method",
    )

    if set_name == "set_H_percentile":
        _write_csv(os.path.join(set_root, "percentile_metrics.csv"), percentile_rows)
        plot_percentile_sweep(
            set_root=set_root,
            rows=percentile_rows,
            metric="avg_auc_clean",
            out_name="percentile_vs_avg_auc_clean.png",
            title="Set H: percentile vs avg_auc_clean",
        )
        plot_percentile_sweep(
            set_root=set_root,
            rows=percentile_rows,
            metric="avg_auc_robust",
            out_name="percentile_vs_avg_auc_robust.png",
            title="Set H: percentile vs avg_auc_robust",
        )
        plot_percentile_sweep(
            set_root=set_root,
            rows=percentile_rows,
            metric="avg_final_clean",
            out_name="percentile_vs_avg_final_clean.png",
            title="Set H: percentile vs avg_final_clean",
        )
        plot_percentile_sweep(
            set_root=set_root,
            rows=percentile_rows,
            metric="avg_final_robust",
            out_name="percentile_vs_avg_final_robust.png",
            title="Set H: percentile vs avg_final_robust",
        )

    write_set_summary_md(set_root=set_root, set_name=set_name, rows=comparison_rows, missing_by_experiment=missing_by_experiment)
    print(f"[aggregate] completed {set_name} ({len(comparison_rows)} experiments)")


if __name__ == "__main__":
    main()

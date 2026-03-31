import copy
import json
import os

import matplotlib.pyplot as plt

from config import parse_config
from runners.active_learning_runner import ActiveLearningRunner
from utils.logger import timestamp_now
from utils.metrics import save_json, save_rows_csv


def resolve_methods(cfg):
    if cfg.run_all_methods:
        return [
            "badge",
            "badge_dual_a",
            "badge_dual_b",
        ]
    if cfg.methods:
        return cfg.methods
    return [cfg.acquisition_method]


def main():
    cfg = parse_config()
    methods = resolve_methods(cfg)

    all_summaries = []
    root_out = os.path.abspath(cfg.output_dir)
    os.makedirs(root_out, exist_ok=True)

    for method in methods:
        method_cfg = copy.deepcopy(cfg)
        method_cfg.acquisition_method = method
        if cfg.run_name is not None and len(methods) > 1:
            method_cfg.run_name = f"{cfg.run_name}_{method}"
        elif cfg.run_name is None:
            method_cfg.run_name = f"{method}_{timestamp_now()}"

        runner = ActiveLearningRunner(method_cfg, method_name=method)
        summary = runner.run()
        all_summaries.append(summary)

    # Build cross-method comparison figure from per-round metrics.
    if len(methods) > 1:
        stamp = timestamp_now()
        fig_path = os.path.join(root_out, f"comparison_figure_{stamp}.png")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_specs = [
            ("clean_acc", "Clean Accuracy (%)"),
            ("fgsm_acc", "FGSM Robust Accuracy (%)"),
            ("round_time_sec", "Round Time (s)"),
        ]

        for method_summary in all_summaries:
            method = method_summary["method"]
            run_dir = method_summary["run_dir"]
            round_json = os.path.join(run_dir, "round_metrics.json")
            if not os.path.exists(round_json):
                continue
            with open(round_json, "r", encoding="utf-8") as f:
                rows = json.load(f)
            budgets = [r["labeled_size"] for r in rows]
            for ax, (key, ylabel) in zip(axes, plot_specs):
                values = [r[key] for r in rows]
                ax.plot(budgets, values, marker="o", label=method)
                ax.set_xlabel("Label Budget")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)

        axes[0].set_title("Label Budget vs Clean Accuracy")
        axes[1].set_title("Label Budget vs FGSM Accuracy")
        axes[2].set_title("Label Budget vs Round Time")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)

        save_json({"comparison_figure": fig_path, "methods": methods}, os.path.join(root_out, f"comparison_figure_{stamp}.json"))

    summary_rows = []
    for item in all_summaries:
        fr = item["final_round"]
        summary_rows.append(
            {
                "method": item["method"],
                "run_dir": item["run_dir"],
                "total_time_sec": item["total_time_sec"],
                "final_round": fr["round"],
                "final_labeled_size": fr["labeled_size"],
                "final_clean_acc": fr["clean_acc"],
                "final_fgsm_acc": fr["fgsm_acc"],
                "final_pgd_acc": fr.get("pgd_acc", fr.get("pgd10_acc", -1.0)),
                "final_avg_logit_mismatch": fr.get("avg_logit_mismatch", -1.0),
                "best_clean_acc": item["best_clean_acc"],
                "best_pgd_acc": item["best_pgd_acc"],
            }
        )

    stamp = timestamp_now()
    save_rows_csv(summary_rows, os.path.join(root_out, f"final_summary_{stamp}.csv"))
    save_json(summary_rows, os.path.join(root_out, f"final_summary_{stamp}.json"))


if __name__ == "__main__":
    main()

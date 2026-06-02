import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
SEEDS = [0, 1, 2, 3, 4]
ALGORITHM_ORDER = ["BADGE", "Entropy", "ours_badge_secant", "ours_badge_secant_p75", "Random", "SAAL"]

SOURCE_MAP = {
    "small_cnn_CIFAR10": {
        "root": os.path.join(EXPERIMENTS_ROOT, "experimentMain", "set_ADV_main_smallcnn_cifar10"),
        "algorithms": {
            "BADGE": "cifar10_small_cnn_badge_b100_r20",
            "Entropy": "cifar10_smallcnn_entropy_b100_r20",
            "ours_badge_secant": "cifar10_small_cnn_ours_badge_secant_b100_r20",
            "ours_badge_secant_p75": "cifar10_small_cnn_ours_badge_secant_p75_b100_r20",
            "Random": "cifar10_smallcnn_random_b100_r20",
            "SAAL": "cifar10_smallcnn_saal_b100_r20",
        },
    },
    "small_cnn_CIFAR100": {
        "root": os.path.join(EXPERIMENTS_ROOT, "experimentMain", "set_ADV_main_smallcnn_cifar100"),
        "algorithms": {
            "BADGE": "cifar100_small_cnn_badge_p10_b100_r20",
            "Entropy": "cifar100_smallcnn_entropy_b100_r20",
            "ours_badge_secant": "cifar100_small_cnn_ours_badge_secant_b100_r20",
            "ours_badge_secant_p75": "cifar100_small_cnn_ours_badge_secant_p75_b100_r20",
            "Random": "cifar100_smallcnn_random_b100_r20",
            "SAAL": "cifar100_smallcnn_saal_b100_r20",
        },
    },
}

TARGET_ROOTS = [
    os.path.join(EXPERIMENTS_ROOT, "experimentA", "init500_acq100_round20"),
    os.path.join(EXPERIMENTS_ROOT, "experimentB", "init500_acq100_round20"),
    os.path.join(EXPERIMENTS_ROOT, "experimentC", "init500_acq100_round20"),
    os.path.join(EXPERIMENTS_ROOT, "experimentD", "init500_acq100_round20_eps1_255"),
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_file_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        _ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)


def _reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _copy_algorithm_subset(src_algo_dir: str, dst_algo_dir: str) -> None:
    _reset_dir(dst_algo_dir)
    logs_src = os.path.join(src_algo_dir, "logs")
    logs_dst = os.path.join(dst_algo_dir, "logs")
    _ensure_dir(logs_dst)

    for seed in SEEDS:
        seed_name = f"seed_{seed}"
        src_seed_dir = os.path.join(src_algo_dir, seed_name)
        dst_seed_dir = os.path.join(dst_algo_dir, seed_name)
        if os.path.isdir(src_seed_dir):
            shutil.copytree(src_seed_dir, dst_seed_dir)
        _copy_file_if_exists(os.path.join(logs_src, f"{seed_name}.log"), os.path.join(logs_dst, f"{seed_name}.log"))


def _read_round_rows(seed_dir: str) -> List[Dict]:
    json_path = os.path.join(seed_dir, "round_metrics.json")
    csv_path = os.path.join(seed_dir, "round_metrics.csv")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return []
        rows = [row for row in rows if isinstance(row, dict)]
    elif os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        return []
    rows.sort(key=lambda row: int(row.get("round", 0)))
    return rows


def _to_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _safe_auc(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    x = np.arange(arr.size, dtype=np.float64)
    return float(np.trapz(arr, x=x) / float(arr.size - 1))


def _mean_std_str(mean: float, std: float) -> str:
    return f"{mean:.2f} +- {std:.2f}"


def _algorithm_display_name(name: str) -> str:
    return {
        "BADGE": "BADGE",
        "Entropy": "Entropy",
        "Random": "Random",
        "SAAL": "SAAL",
        "ours_badge_secant": "Ours BADGE Secant",
        "ours_badge_secant_p75": "Ours BADGE Secant p75",
    }[name]


def _collect_algorithm_summary(algorithm_dir: str) -> Dict:
    seed_rows = {}
    metrics = {
        "clean_final": [],
        "clean_auc": [],
        "fgsm_final": [],
        "fgsm_auc": [],
        "pgd_final": [],
        "pgd_auc": [],
        "round_time": [],
        "train_time": [],
        "eval_time": [],
        "acq_score": [],
        "selection": [],
    }
    for seed in SEEDS:
        seed_dir = os.path.join(algorithm_dir, f"seed_{seed}")
        rows = _read_round_rows(seed_dir)
        if not rows:
            seed_rows[seed] = {"status": "missing", "last_round": None, "rows": 0}
            continue
        clean = [_to_float(row.get("clean_acc")) for row in rows]
        fgsm = [_to_float(row.get("fgsm_acc")) for row in rows]
        pgd = [_to_float(row.get("pgd_acc", row.get("pgd10_acc"))) for row in rows]
        round_time = [_to_float(row.get("round_time_sec")) for row in rows]
        train_time = [_to_float(row.get("train_time_sec")) for row in rows]
        eval_time = [_to_float(row.get("eval_time_sec")) for row in rows]
        acq_score = [_to_float(row.get("acquisition_scoring_time_sec")) for row in rows[1:]]
        selection = [_to_float(row.get("selection_time_sec")) for row in rows[1:]]

        metrics["clean_final"].append(clean[-1])
        metrics["clean_auc"].append(_safe_auc(clean))
        metrics["fgsm_final"].append(fgsm[-1])
        metrics["fgsm_auc"].append(_safe_auc(fgsm))
        metrics["pgd_final"].append(pgd[-1])
        metrics["pgd_auc"].append(_safe_auc(pgd))
        metrics["round_time"].append(float(np.nanmean(np.asarray(round_time, dtype=np.float64))))
        metrics["train_time"].append(float(np.nanmean(np.asarray(train_time, dtype=np.float64))))
        metrics["eval_time"].append(float(np.nanmean(np.asarray(eval_time, dtype=np.float64))))
        metrics["acq_score"].append(float(np.nanmean(np.asarray(acq_score, dtype=np.float64))) if acq_score else 0.0)
        metrics["selection"].append(float(np.nanmean(np.asarray(selection, dtype=np.float64))) if selection else 0.0)
        seed_rows[seed] = {"status": "complete", "last_round": int(rows[-1].get("round", 0)), "rows": len(rows)}

    def _agg(vals: List[float]) -> Tuple[float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    return {
        "seed_rows": seed_rows,
        "completed": sum(1 for v in seed_rows.values() if v["status"] == "complete"),
        "metrics": {k: _agg(v) for k, v in metrics.items()},
    }


def _infer_setup_text(setting_name: str) -> str:
    init = "500" if "init500" in setting_name else "unknown"
    acq = "100" if "acq100" in setting_name else "unknown"
    rounds = "20" if "round20" in setting_name else "unknown"
    if "eps1_255" in setting_name:
        epsilon = "1/255"
    else:
        epsilon = "1/255 (default)"
    return f"init={init}, acq={acq}, rounds={rounds}, epsilon={epsilon}"


def _write_analysis_bundle(dataset_dir: str) -> None:
    setting_name = os.path.basename(os.path.dirname(dataset_dir))
    experiment_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))
    dataset_name = os.path.basename(dataset_dir)
    model_token = "smallcnn"
    dataset_token = "cifar10" if dataset_name.endswith("CIFAR10") else "cifar100"
    title_dataset = "CIFAR-10" if dataset_token == "cifar10" else "CIFAR-100"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S KST")

    analysis_dir = os.path.join(dataset_dir, "analysis", f"analysis_time_{timestamp}_root_algorithms_no_hauc_no_batchbald_stats")
    _reset_dir(analysis_dir)

    summaries = []
    for algorithm in ALGORITHM_ORDER:
        algo_dir = os.path.join(dataset_dir, algorithm)
        summary = _collect_algorithm_summary(algo_dir)
        summaries.append((algorithm, summary))

    csv_rows = []
    for algorithm, summary in summaries:
        m = summary["metrics"]
        csv_rows.append(
            {
                "Algorithm": _algorithm_display_name(algorithm),
                "completed seeds": f"{summary['completed']}/{len(SEEDS)}",
                "Clean Final mean": f"{m['clean_final'][0]:.2f}",
                "Clean Final std": f"{m['clean_final'][1]:.2f}",
                "Clean AUC mean": f"{m['clean_auc'][0]:.2f}",
                "Clean AUC std": f"{m['clean_auc'][1]:.2f}",
                "FGSM Final mean": f"{m['fgsm_final'][0]:.2f}",
                "FGSM Final std": f"{m['fgsm_final'][1]:.2f}",
                "FGSM AUC mean": f"{m['fgsm_auc'][0]:.2f}",
                "FGSM AUC std": f"{m['fgsm_auc'][1]:.2f}",
                "PGD Final mean": f"{m['pgd_final'][0]:.2f}",
                "PGD Final std": f"{m['pgd_final'][1]:.2f}",
                "PGD AUC mean": f"{m['pgd_auc'][0]:.2f}",
                "PGD AUC std": f"{m['pgd_auc'][1]:.2f}",
                "Round Time / round mean": f"{m['round_time'][0]:.2f}",
                "Round Time / round std": f"{m['round_time'][1]:.2f}",
                "Train Time / round mean": f"{m['train_time'][0]:.2f}",
                "Train Time / round std": f"{m['train_time'][1]:.2f}",
                "Eval Time / round mean": f"{m['eval_time'][0]:.2f}",
                "Eval Time / round std": f"{m['eval_time'][1]:.2f}",
                "Acq Score / acq round mean": f"{m['acq_score'][0]:.2f}",
                "Acq Score / acq round std": f"{m['acq_score'][1]:.2f}",
                "Selection / acq round mean": f"{m['selection'][0]:.2f}",
                "Selection / acq round std": f"{m['selection'][1]:.2f}",
                "Experiment Directory": os.path.relpath(os.path.join(dataset_dir, algorithm), PROJECT_ROOT).replace("\\", "/"),
            }
        )

    csv_name = f"{model_token}_{dataset_token}_{setting_name}_root_algorithms_no_hauc_no_batchbald_stats_summary.csv"
    csv_path = os.path.join(analysis_dir, csv_name)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    best_clean = max(csv_rows, key=lambda r: float(r["Clean Final mean"]))["Algorithm"]
    best_clean_auc = max(csv_rows, key=lambda r: float(r["Clean AUC mean"]))["Algorithm"]
    best_fgsm = max(csv_rows, key=lambda r: float(r["FGSM Final mean"]))["Algorithm"]
    best_fgsm_auc = max(csv_rows, key=lambda r: float(r["FGSM AUC mean"]))["Algorithm"]
    best_pgd = max(csv_rows, key=lambda r: float(r["PGD Final mean"]))["Algorithm"]
    best_pgd_auc = max(csv_rows, key=lambda r: float(r["PGD AUC mean"]))["Algorithm"]

    def _fmt_td(val: str, is_best: bool) -> str:
        if is_best:
            return f"<font color=\"blue\"><b>{val}</b></font>"
        return val

    lines = [
        f"# SmallCNN {title_dataset} Root-Level Algorithm Statistics — {experiment_name} / {setting_name}",
        "",
        f"- Analysis time: `{time_label}`",
        f"- Scope: `experiments/{experiment_name}/{setting_name}/{dataset_name}`",
        "- Model: `small_cnn`",
        f"- Dataset: `{dataset_token}`",
        f"- Setup: `{_infer_setup_text(setting_name)}`",
        "- Algorithms: `BADGE, Entropy, Random, SAAL, Ours BADGE Secant, Ours BADGE Secant p75`",
        "- Excluded: `anothers/` subtree and `BatchBALD`",
        "- Completed seed criterion: final round >= `20`",
        "- AUC: normalized trapezoidal area over rounds 0..20",
        "- Std: population std across completed seeds",
        "- Blue cells indicate the best mean value in that column.",
        "",
        "## Main Table",
        "",
        "<table>",
        "  <thead>",
        "    <tr><th>Algorithm</th><th>completed seeds</th><th>Clean Final</th><th>Clean AUC</th><th>FGSM Final</th><th>FGSM AUC</th><th>PGD Final</th><th>PGD AUC</th></tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in csv_rows:
        lines.append(
            "    <tr>"
            f"<td>{row['Algorithm']}</td>"
            f"<td>{row['completed seeds']}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['Clean Final mean']), float(row['Clean Final std'])), row['Algorithm'] == best_clean)}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['Clean AUC mean']), float(row['Clean AUC std'])), row['Algorithm'] == best_clean_auc)}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['FGSM Final mean']), float(row['FGSM Final std'])), row['Algorithm'] == best_fgsm)}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['FGSM AUC mean']), float(row['FGSM AUC std'])), row['Algorithm'] == best_fgsm_auc)}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['PGD Final mean']), float(row['PGD Final std'])), row['Algorithm'] == best_pgd)}</td>"
            f"<td>{_fmt_td(_mean_std_str(float(row['PGD AUC mean']), float(row['PGD AUC std'])), row['Algorithm'] == best_pgd_auc)}</td>"
            "</tr>"
        )
    lines += [
        "  </tbody>",
        "</table>",
        "",
        "## Timing Table",
        "",
        "<table>",
        "  <thead>",
        "    <tr><th>Algorithm</th><th>completed seeds</th><th>Round Time / round</th><th>Train Time / round</th><th>Eval Time / round</th><th>Acq Score / acq round</th><th>Selection / acq round</th></tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in csv_rows:
        lines.append(
            "    <tr>"
            f"<td>{row['Algorithm']}</td>"
            f"<td>{row['completed seeds']}</td>"
            f"<td>{_mean_std_str(float(row['Round Time / round mean']), float(row['Round Time / round std']))}</td>"
            f"<td>{_mean_std_str(float(row['Train Time / round mean']), float(row['Train Time / round std']))}</td>"
            f"<td>{_mean_std_str(float(row['Eval Time / round mean']), float(row['Eval Time / round std']))}</td>"
            f"<td>{_mean_std_str(float(row['Acq Score / acq round mean']), float(row['Acq Score / acq round std']))}</td>"
            f"<td>{_mean_std_str(float(row['Selection / acq round mean']), float(row['Selection / acq round std']))}</td>"
            "</tr>"
        )
    lines += [
        "  </tbody>",
        "</table>",
        "",
        "## Seed Completion Status",
        "",
        "| Algorithm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for algorithm, summary in summaries:
        cells = []
        for seed in SEEDS:
            row = summary["seed_rows"][seed]
            if row["status"] == "complete":
                cells.append(f"complete (last round {row['last_round']}, rows {row['rows']})")
            else:
                cells.append("missing")
        lines.append(f"| {_algorithm_display_name(algorithm)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Experiment Directories",
        "",
    ]
    for algorithm in ALGORITHM_ORDER:
        lines.append(
            f"- {_algorithm_display_name(algorithm)}: `experiments/{experiment_name}/{setting_name}/{dataset_name}/{algorithm}`"
        )

    md_name = f"{model_token}_{dataset_token}_{setting_name}_root_algorithms_no_hauc_no_batchbald_stats_summary.md"
    md_path = os.path.join(analysis_dir, md_name)
    preview_path = os.path.join(
        analysis_dir,
        f"{model_token}_{dataset_token}_{setting_name}_root_algorithms_no_hauc_no_batchbald_stats_summary_preview.md",
    )
    content = "\n".join(lines) + "\n"
    for path in (md_path, preview_path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)


def _populate_target_root(target_root: str) -> None:
    for dataset_dirname, source_cfg in SOURCE_MAP.items():
        dst_dataset_dir = os.path.join(target_root, dataset_dirname)
        _ensure_dir(dst_dataset_dir)
        for algorithm in ALGORITHM_ORDER:
            src_dir = os.path.join(source_cfg["root"], source_cfg["algorithms"][algorithm])
            dst_dir = os.path.join(dst_dataset_dir, algorithm)
            _copy_algorithm_subset(src_dir, dst_dir)
        _write_analysis_bundle(dst_dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate selected smallcnn seed0-4 subsets into experimentA-D targets.")
    parser.add_argument("--graphs", action="store_true", help="Regenerate _graphs for experimentA-D after population.")
    args = parser.parse_args()

    for target_root in TARGET_ROOTS:
        _populate_target_root(target_root)
        print(f"[populate] done: {target_root}")

    if args.graphs:
        script_path = os.path.join(PROJECT_ROOT, "scripts", "build_experimentA_E_graphs.py")
        cmd = f"python {script_path} --experiments experimentA,experimentB,experimentC,experimentD"
        raise SystemExit(os.system(cmd))


if __name__ == "__main__":
    main()

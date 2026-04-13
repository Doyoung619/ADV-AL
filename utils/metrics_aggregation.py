import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def _robust_from_row(row: Dict) -> float:
    for key in ("pgd_acc", "pgd10_acc", "pgd_robust_acc", "fgsm_acc"):
        if key in row:
            return float(row[key])
    return float("nan")


def _safe_auc(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    x = np.arange(arr.size, dtype=np.float64)
    auc = np.trapz(arr, x=x)
    return float(auc / float(arr.size - 1))


def read_seed_round_rows(seed_dir: str) -> List[Dict]:
    path = os.path.join(seed_dir, "round_metrics.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return []
    rows = [r for r in rows if isinstance(r, dict)]
    rows.sort(key=lambda r: int(r.get("round", 0)))
    return rows


def compute_seed_summary(seed_dir: str) -> Dict[str, float]:
    rows = read_seed_round_rows(seed_dir)
    if not rows:
        return {}
    clean_curve = [float(r.get("clean_acc", float("nan"))) for r in rows]
    robust_curve = [_robust_from_row(r) for r in rows]
    out = {
        "avg_auc_clean": _safe_auc(clean_curve),
        "avg_auc_robust": _safe_auc(robust_curve),
        "avg_final_clean": float(clean_curve[-1]),
        "avg_final_robust": float(robust_curve[-1]),
        "avg_best_clean": float(np.nanmax(np.asarray(clean_curve, dtype=np.float64))),
        "avg_best_robust": float(np.nanmax(np.asarray(robust_curve, dtype=np.float64))),
        "num_round_rows": float(len(rows)),
    }

    # Runtime-related extras.
    acq_times = [float(r.get("acquisition_scoring_time_sec", float("nan"))) for r in rows]
    round_times = [float(r.get("round_time_sec", float("nan"))) for r in rows]
    out["avg_acquisition_scoring_time_sec"] = float(np.nanmean(np.asarray(acq_times, dtype=np.float64)))
    out["avg_round_time_sec"] = float(np.nanmean(np.asarray(round_times, dtype=np.float64)))

    summary_path = os.path.join(seed_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
            out["total_time_sec"] = float(sm.get("total_time_sec", float("nan")))
        except Exception:
            out["total_time_sec"] = float("nan")
    else:
        out["total_time_sec"] = float("nan")

    return out


def aggregate_seed_metrics(seed_metrics: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = set()
    for m in seed_metrics.values():
        keys.update(m.keys())
    agg: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        vals = []
        for m in seed_metrics.values():
            v = m.get(key, float("nan"))
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                vals.append(float(v))
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0.0}
        else:
            agg[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": float(arr.size),
            }
    return agg


def collect_curves(exp_dir: str, seed_list: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    clean_curves = []
    robust_curves = []
    available = []
    for seed in seed_list:
        seed_dir = os.path.join(exp_dir, f"seed_{seed}")
        rows = read_seed_round_rows(seed_dir)
        if not rows:
            continue
        clean_curves.append(np.asarray([float(r.get("clean_acc", float("nan"))) for r in rows], dtype=np.float64))
        robust_curves.append(np.asarray([_robust_from_row(r) for r in rows], dtype=np.float64))
        available.append(seed)
    if not clean_curves:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64), []
    min_len = min(c.shape[0] for c in clean_curves)
    clean = np.stack([c[:min_len] for c in clean_curves], axis=0)
    robust = np.stack([c[:min_len] for c in robust_curves], axis=0)
    return clean, robust, available


def summarize_experiment_dir(exp_dir: str, seed_list: List[int]) -> Dict:
    seed_metrics: Dict[int, Dict[str, float]] = {}
    missing = []
    for seed in seed_list:
        seed_dir = os.path.join(exp_dir, f"seed_{seed}")
        metrics = compute_seed_summary(seed_dir)
        if not metrics:
            missing.append(seed)
            continue
        seed_metrics[seed] = metrics

    aggregate = aggregate_seed_metrics(seed_metrics)
    clean_curves, robust_curves, available = collect_curves(exp_dir, seed_list)
    return {
        "seed_metrics": seed_metrics,
        "aggregate": aggregate,
        "missing_seeds": missing,
        "available_seeds": available,
        "clean_curves": clean_curves,
        "robust_curves": robust_curves,
    }

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from acquisition.bald import score_bald
from acquisition.entropy import score_entropy
from acquisition.ours import score_ours, score_ours_gap, score_ours_hessian
from datasets import build_loader, full_train_indices, get_cifar10_transforms, load_cifar10
from models import build_model
from train import train_model_for_round
from utils.feature_extractor import extract_features_probs
from utils.logger import ProgressLogger, setup_logger, timestamp_now
from utils.seed import set_seed


def _parse_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_candidate_indices(unlabeled_idx: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size <= 0 or size >= len(unlabeled_idx):
        return np.sort(unlabeled_idx.astype(np.int64))
    rng = np.random.default_rng(seed)
    picked = rng.choice(unlabeled_idx, size=size, replace=False).astype(np.int64)
    return np.sort(picked)


def _quantile_threshold(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(np.float64), q=q))


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


def _compute_outlier_scores(features: torch.Tensor, k: int, chunk_size: int, device: torch.device) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got {tuple(features.shape)}")
    n = int(features.size(0))
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)
    k_eff = max(1, min(int(k), n - 1))

    feats = features.to(device=device, dtype=torch.float32)
    out = torch.empty(n, dtype=torch.float32, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = feats[start:end]
        dists = torch.cdist(chunk, feats, p=2.0)  # [B, N]
        row_idx = torch.arange(start, end, device=device)
        dists[torch.arange(end - start, device=device), row_idx] = float("inf")
        knn_vals = torch.topk(dists, k=k_eff, largest=False, dim=1).values
        out[start:end] = knn_vals.mean(dim=1)

    return out.detach().cpu().numpy()


def _compute_ours_scores(args, model, candidate_loader, device: torch.device, progress: ProgressLogger) -> np.ndarray:
    method = args.ours_method.lower()
    if method in {"ours", "ours_l2"}:
        scores = score_ours(
            model=model,
            unlabeled_loader=candidate_loader,
            device=device,
            epsilon=args.epsilon,
            mean=args.cifar10_mean,
            std=args.cifar10_std,
            attack=args.acquisition_attack,
            pgd_steps=args.acquisition_pgd_steps,
            pgd_alpha=args.acquisition_pgd_alpha,
            delta_objective=args.ours_delta_objective,
            progress_logger=progress,
        )
    elif method == "ours_hessian":
        scores = score_ours_hessian(
            model=model,
            unlabeled_loader=candidate_loader,
            device=device,
            epsilon=args.epsilon,
            mean=args.cifar10_mean,
            std=args.cifar10_std,
            attack=args.acquisition_attack,
            pgd_steps=args.acquisition_pgd_steps,
            pgd_alpha=args.acquisition_pgd_alpha,
            delta_objective=args.ours_delta_objective,
            hessian_lambda=args.ours_hessian_lambda,
            progress_logger=progress,
        )
    elif method == "ours_gap":
        scores = score_ours_gap(
            model=model,
            unlabeled_loader=candidate_loader,
            device=device,
            epsilon=args.epsilon,
            mean=args.cifar10_mean,
            std=args.cifar10_std,
            pgd_steps=args.acquisition_pgd_steps,
            pgd_alpha=args.acquisition_pgd_alpha,
            use_fixed_clean_classes=args.ours_gap_use_fixed_clean_classes,
            progress_logger=progress,
        )
    else:
        raise ValueError(f"Unsupported ours_method: {args.ours_method}")
    return scores.detach().cpu().numpy()


def _compute_uncertainty_scores(args, model, candidate_loader, device: torch.device, progress: ProgressLogger) -> np.ndarray:
    method = args.uncertainty_method.lower()
    if method == "bald":
        scores = score_bald(
            model=model,
            unlabeled_loader=candidate_loader,
            mc_passes=args.mc_passes,
            device=device,
            progress_logger=progress,
        )
    elif method == "entropy":
        scores = score_entropy(
            model=model,
            unlabeled_loader=candidate_loader,
            device=device,
            use_mc=False,
            progress_logger=progress,
        )
    else:
        raise ValueError(f"Unsupported uncertainty_method: {args.uncertainty_method}")
    return scores.detach().cpu().numpy()


def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    k_eff = min(max(0, int(k)), n)
    mask = np.zeros(n, dtype=np.int64)
    if k_eff == 0:
        return mask
    picked = np.argpartition(scores, -k_eff)[-k_eff:]
    mask[picked] = 1
    return mask


def _topk_mask_with_allowed(scores: np.ndarray, allowed_mask: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    mask = np.zeros(n, dtype=np.int64)
    allowed_idx = np.flatnonzero(allowed_mask.astype(bool))
    if allowed_idx.size == 0:
        return mask
    k_eff = min(max(0, int(k)), int(allowed_idx.size))
    if k_eff == 0:
        return mask
    allowed_scores = scores[allowed_idx]
    local = np.argpartition(allowed_scores, -k_eff)[-k_eff:]
    picked = allowed_idx[local]
    mask[picked] = 1
    return mask


def _save_analysis_csv(path: str, rows: Dict[str, np.ndarray]) -> None:
    columns = [
        "sample_index",
        "our_score",
        "uncertainty_score",
        "outlier_score",
        "selected_pure",
        "selected_filtered",
    ]
    n = len(rows["sample_index"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n):
            writer.writerow(
                [
                    int(rows["sample_index"][i]),
                    float(rows["our_score"][i]),
                    float(rows["uncertainty_score"][i]),
                    float(rows["outlier_score"][i]),
                    int(rows["selected_pure"][i]),
                    int(rows["selected_filtered"][i]),
                ]
            )


def _make_plots(
    out_dir: str,
    our_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
    outlier_scores: np.ndarray,
    pure_mask: np.ndarray,
    filtered_mask: np.ndarray,
    uncertainty_cutoff: float,
) -> Tuple[str, str]:
    fig_a = os.path.join(out_dir, "figure_a_ours_vs_outlier.png")
    fig_b = os.path.join(out_dir, "figure_b_outlier_vs_uncertainty.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(our_scores, outlier_scores, s=10, c="gray", alpha=0.25, label="all candidates")
    plt.scatter(our_scores[pure_mask == 1], outlier_scores[pure_mask == 1], s=18, c="red", alpha=0.8, label="pure ours")
    plt.scatter(
        our_scores[filtered_mask == 1],
        outlier_scores[filtered_mask == 1],
        s=18,
        c="blue",
        alpha=0.8,
        label="filtered ours",
    )
    plt.xlabel("Our Score")
    plt.ylabel("Outlier-like Score (kNN avg dist)")
    plt.title("Our score vs. outlier-likeness")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_a, dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(outlier_scores, uncertainty_scores, s=10, c="gray", alpha=0.25, label="all candidates")
    plt.scatter(
        outlier_scores[pure_mask == 1],
        uncertainty_scores[pure_mask == 1],
        s=18,
        c="red",
        alpha=0.8,
        label="pure ours",
    )
    plt.scatter(
        outlier_scores[filtered_mask == 1],
        uncertainty_scores[filtered_mask == 1],
        s=18,
        c="blue",
        alpha=0.8,
        label="filtered ours",
    )
    plt.axhline(uncertainty_cutoff, linestyle="--", linewidth=1.2, color="black", alpha=0.8, label="uncertainty cutoff")
    plt.xlabel("Outlier-like Score (kNN avg dist)")
    plt.ylabel("Uncertainty Score")
    plt.title("Outlier-likeness vs. uncertainty")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_b, dpi=220)
    plt.close()

    return fig_a, fig_b


@dataclass
class AnalysisSummary:
    run_dir: str
    seed: int
    candidate_pool_size: int
    acquisition_size: int
    filter_percentile: float
    uncertainty_method: str
    ours_method: str
    uncertainty_cutoff: float
    spearman_ours_vs_outlier: float
    spearman_outlier_vs_uncertainty: float
    mean_outlier_all: float
    mean_outlier_pure: float
    mean_outlier_filtered: float
    mean_uncertainty_all: float
    mean_uncertainty_pure: float
    mean_uncertainty_filtered: float
    top_outlier_percent: float
    top_outlier_removed_fraction: float
    bad_region_pure_fraction: float
    bad_region_filtered_fraction: float
    figure_a_path: str
    figure_b_path: str
    analysis_csv_path: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-shot diagnostic analysis: ours score / uncertainty / outlierness.")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--download-if-missing", action="store_true")
    p.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "resnet10", "resnet18"])
    p.add_argument("--dropout-p", type=float, default=0.2)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.set_defaults(channels_last=True)

    p.add_argument("--output-dir", type=str, default="./output_Experiment2")
    p.add_argument("--run-name", type=str, default=None)

    p.add_argument("--initial-labeled-size", type=int, default=500)
    p.add_argument("--acquisition-size", type=int, default=200)
    p.add_argument("--epochs-per-round", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--pool-batch-size", type=int, default=256)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    p.set_defaults(save_checkpoints=True)

    p.add_argument("--ours-method", type=str, default="ours_l2", choices=["ours", "ours_l2", "ours_hessian", "ours_gap"])
    p.add_argument("--uncertainty-method", type=str, default="bald", choices=["bald", "entropy"])
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--filter-percentile", type=float, default=10.0)
    p.add_argument("--analysis-candidate-pool-size", type=int, default=5000)
    p.add_argument("--outlier-k", type=int, default=10)
    p.add_argument("--knn-chunk-size", type=int, default=512)
    p.add_argument("--top-outlier-percent", type=float, default=10.0)

    p.add_argument("--epsilon", type=float, default=1.0 / 255.0)
    p.add_argument("--acquisition-attack", type=str, default="pgd", choices=["fgsm", "pgd"])
    p.add_argument("--acquisition-pgd-steps", type=int, default=5)
    p.add_argument("--acquisition-pgd-alpha", type=float, default=2.0 / 255.0)
    p.add_argument("--ours-delta-objective", type=str, default="logit_mismatch", choices=["logit_mismatch", "predictive_ce"])
    p.add_argument("--ours-hessian-lambda", type=float, default=1e-3)
    p.add_argument("--ours-gap-use-fixed-clean-classes", action="store_true")
    p.add_argument("--no-ours-gap-use-fixed-clean-classes", dest="ours_gap_use_fixed_clean_classes", action="store_false")
    p.set_defaults(ours_gap_use_fixed_clean_classes=True)

    return p


def main():
    args = build_parser().parse_args()
    if not (0.0 <= args.filter_percentile < 100.0):
        raise ValueError(f"filter_percentile must be in [0, 100), got {args.filter_percentile}")
    if not (0.0 < args.top_outlier_percent <= 100.0):
        raise ValueError(f"top_outlier_percent must be in (0, 100], got {args.top_outlier_percent}")
    if args.initial_labeled_size <= 0:
        raise ValueError("initial_labeled_size must be positive.")

    args.cifar10_mean = (0.4914, 0.4822, 0.4465)
    args.cifar10_std = (0.2023, 0.1994, 0.2010)

    device = _parse_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    run_name = args.run_name or f"diagnostic_{args.ours_method}_{args.uncertainty_method}_p{int(args.filter_percentile)}_{timestamp_now()}"
    run_dir = os.path.abspath(os.path.join(args.output_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "analysis.log"), name=f"diagnostic_{run_name}")
    progress = ProgressLogger(logger)
    progress.log(
        (
            f"Starting diagnostic run={run_name} seed={args.seed} device={device} "
            f"ours={args.ours_method} uncertainty={args.uncertainty_method} "
            f"filter_p={args.filter_percentile} candidate_pool_size={args.analysis_candidate_pool_size}"
        ),
        device=str(device),
    )
    progress.log(
        (
            f"Acquisition score config: epsilon={args.epsilon:.10f} "
            f"attack={args.acquisition_attack} pgd_steps={args.acquisition_pgd_steps} "
            f"pgd_alpha={args.acquisition_pgd_alpha:.10f}"
        ),
        device=str(device),
    )

    train_base, _ = load_cifar10(args.data_dir, download_if_missing=args.download_if_missing)
    train_tf, eval_tf = get_cifar10_transforms(args.cifar10_mean, args.cifar10_std)
    full_idx = full_train_indices(train_base)

    rng = np.random.default_rng(args.seed)
    labeled_idx = np.sort(rng.choice(full_idx, size=args.initial_labeled_size, replace=False).astype(np.int64))
    unlabeled_idx = np.setdiff1d(full_idx, labeled_idx, assume_unique=False)
    candidate_idx = _pick_candidate_indices(
        unlabeled_idx=unlabeled_idx,
        size=int(args.analysis_candidate_pool_size),
        seed=args.seed * 97 + 11,
    )

    progress.log(
        (
            f"Split prepared labeled={len(labeled_idx)} unlabeled={len(unlabeled_idx)} "
            f"analysis_candidates={len(candidate_idx)}"
        ),
        device=str(device),
    )

    model = build_model(model_name=args.model, num_classes=args.num_classes, dropout_p=args.dropout_p)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    train_loader = build_loader(
        base_dataset=train_base,
        indices=labeled_idx,
        transform=train_tf,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    train_cfg = SimpleNamespace(
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        epochs_per_round=args.epochs_per_round,
        min_lr=args.min_lr,
        amp=bool(args.amp),
        channels_last=bool(args.channels_last),
        save_checkpoints=bool(args.save_checkpoints),
    )

    round0_dir = os.path.join(run_dir, "round_00_train")
    os.makedirs(round0_dir, exist_ok=True)
    model, train_meta = train_model_for_round(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        cfg=train_cfg,
        device=device,
        round_idx=0,
        round_dir=round0_dir,
        progress_logger=progress,
    )
    with open(os.path.join(run_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    candidate_loader = build_loader(
        base_dataset=train_base,
        indices=candidate_idx,
        transform=eval_tf,
        batch_size=args.pool_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    progress.log("Computing our acquisition scores on analysis candidate pool.", device=str(device))
    our_scores = _compute_ours_scores(args, model, candidate_loader, device, progress)
    progress.log(
        (
            f"Our-score stats: min={float(np.min(our_scores)):.8f} "
            f"max={float(np.max(our_scores)):.8f} "
            f"mean={float(np.mean(our_scores)):.8f} std={float(np.std(our_scores)):.8f}"
        ),
        device=str(device),
    )
    if float(np.max(our_scores) - np.min(our_scores)) < 1e-12:
        progress.log(
            (
                "WARNING: our scores are nearly constant. "
                "If using ours_l2 with logit_mismatch + fgsm, score collapse can happen; "
                "prefer --acquisition-attack pgd."
            ),
            device=str(device),
        )

    progress.log("Computing uncertainty scores on analysis candidate pool.", device=str(device))
    uncertainty_scores = _compute_uncertainty_scores(args, model, candidate_loader, device, progress)

    progress.log("Extracting features and computing kNN outlier-like scores.", device=str(device))
    features, _, indices_tensor = extract_features_probs(model, candidate_loader, device=device)
    order_ok = np.array_equal(indices_tensor.cpu().numpy().astype(np.int64), candidate_idx.astype(np.int64))
    if not order_ok:
        raise RuntimeError("Candidate index order mismatch between score and feature extraction.")
    outlier_scores = _compute_outlier_scores(
        features=features,
        k=args.outlier_k,
        chunk_size=args.knn_chunk_size,
        device=device,
    )

    n = len(candidate_idx)
    k_select = min(int(args.acquisition_size), n)
    pure_mask = _topk_mask(our_scores, k_select)

    filter_q = args.filter_percentile / 100.0
    uncertainty_cutoff = _quantile_threshold(uncertainty_scores, q=filter_q)
    keep_mask = uncertainty_scores >= uncertainty_cutoff
    filtered_mask = _topk_mask_with_allowed(our_scores, keep_mask, k_select)

    analysis_csv_path = os.path.join(run_dir, "analysis_table.csv")
    _save_analysis_csv(
        analysis_csv_path,
        rows={
            "sample_index": candidate_idx,
            "our_score": our_scores,
            "uncertainty_score": uncertainty_scores,
            "outlier_score": outlier_scores,
            "selected_pure": pure_mask,
            "selected_filtered": filtered_mask,
        },
    )

    figure_a_path, figure_b_path = _make_plots(
        out_dir=run_dir,
        our_scores=our_scores,
        uncertainty_scores=uncertainty_scores,
        outlier_scores=outlier_scores,
        pure_mask=pure_mask,
        filtered_mask=filtered_mask,
        uncertainty_cutoff=uncertainty_cutoff,
    )

    pure_sel = pure_mask == 1
    filt_sel = filtered_mask == 1

    top_outlier_q = 1.0 - (args.top_outlier_percent / 100.0)
    top_outlier_cutoff = _quantile_threshold(outlier_scores, q=top_outlier_q)
    top_outlier_mask = outlier_scores >= top_outlier_cutoff
    removed_mask = ~keep_mask
    if top_outlier_mask.sum() > 0:
        top_outlier_removed_fraction = float((top_outlier_mask & removed_mask).sum() / top_outlier_mask.sum())
    else:
        top_outlier_removed_fraction = float("nan")

    ours_top10_cutoff = _quantile_threshold(our_scores, q=0.9)
    unc_bottom50_cutoff = _quantile_threshold(uncertainty_scores, q=0.5)
    bad_region_mask = (our_scores >= ours_top10_cutoff) & (uncertainty_scores <= unc_bottom50_cutoff)
    bad_region_pure_fraction = float(bad_region_mask[pure_sel].mean()) if pure_sel.any() else float("nan")
    bad_region_filtered_fraction = float(bad_region_mask[filt_sel].mean()) if filt_sel.any() else float("nan")

    summary = AnalysisSummary(
        run_dir=run_dir,
        seed=int(args.seed),
        candidate_pool_size=int(n),
        acquisition_size=int(k_select),
        filter_percentile=float(args.filter_percentile),
        uncertainty_method=args.uncertainty_method,
        ours_method=args.ours_method,
        uncertainty_cutoff=float(uncertainty_cutoff),
        spearman_ours_vs_outlier=_spearman_corr(our_scores, outlier_scores),
        spearman_outlier_vs_uncertainty=_spearman_corr(outlier_scores, uncertainty_scores),
        mean_outlier_all=float(outlier_scores.mean()),
        mean_outlier_pure=float(outlier_scores[pure_sel].mean()) if pure_sel.any() else float("nan"),
        mean_outlier_filtered=float(outlier_scores[filt_sel].mean()) if filt_sel.any() else float("nan"),
        mean_uncertainty_all=float(uncertainty_scores.mean()),
        mean_uncertainty_pure=float(uncertainty_scores[pure_sel].mean()) if pure_sel.any() else float("nan"),
        mean_uncertainty_filtered=float(uncertainty_scores[filt_sel].mean()) if filt_sel.any() else float("nan"),
        top_outlier_percent=float(args.top_outlier_percent),
        top_outlier_removed_fraction=top_outlier_removed_fraction,
        bad_region_pure_fraction=bad_region_pure_fraction,
        bad_region_filtered_fraction=bad_region_filtered_fraction,
        figure_a_path=figure_a_path,
        figure_b_path=figure_b_path,
        analysis_csv_path=analysis_csv_path,
    )

    summary_path = os.path.join(run_dir, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    summary_csv_path = os.path.join(run_dir, "summary_stats.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in asdict(summary).items():
            writer.writerow([k, v])

    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    progress.log(f"Diagnostic analysis completed. Outputs saved to: {run_dir}", device=str(device))
    progress.log(f"Figure A: {figure_a_path}", device=str(device))
    progress.log(f"Figure B: {figure_b_path}", device=str(device))
    progress.log(f"Table CSV: {analysis_csv_path}", device=str(device))
    progress.log(f"Summary JSON: {summary_path}", device=str(device))


if __name__ == "__main__":
    main()

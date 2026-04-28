import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from acquisition.logdet_refine import build_gram_matrix, forward_greedy_select, refine_by_swap
from acquisition.secant_badge import _secant_attack_settings, compute_secant_badge_embeddings
from acquisition.utils import AcquisitionOutput, BaseAcquisition, tensor_stats


def compute_secant_norm_scores(embeddings: torch.Tensor) -> torch.Tensor:
    """D(x)=||phi(x)||_2 for secant embeddings phi."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N,D], got shape={tuple(embeddings.shape)}")
    return embeddings.float().norm(dim=1)


def compute_clean_grad_norm_scores(g_clean: torch.Tensor) -> torch.Tensor:
    """D_clean(x)=||g_clean(x)||_2 for clean last-layer pseudo-gradient embeddings."""
    if g_clean.ndim != 2:
        raise ValueError(f"g_clean must be [N,D], got shape={tuple(g_clean.shape)}")
    return g_clean.float().norm(dim=1)


def _scalar_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    xf = x.float().detach().cpu()
    return {
        "min": float(xf.min().item()),
        "median": float(torch.median(xf).item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
    }


def _prefilter_by_scores(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    drop_percent: float,
    budget: int,
    metric: str,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Drop the bottom p% by a scalar score and preserve the original candidate
    order among retained examples. Convention: keep ceil((1-p/100)N), so p10
    keeps ceil(0.9N) and drops floor(0.1N), while always keeping at least budget.
    """
    if not (0.0 <= float(drop_percent) <= 100.0):
        raise ValueError(f"drop_percent must be in [0,100], got {drop_percent}")
    n = int(embeddings.size(0))
    if scores.ndim != 1:
        raise ValueError(f"scores must be [N], got shape={tuple(scores.shape)}")
    if int(scores.numel()) != n:
        raise ValueError(f"scores length must match embeddings rows, got {scores.numel()} vs {n}")
    scores = scores.float()
    stats_key = f"{metric}_stats"
    retained_stats_key = f"{metric}_retained_stats"
    if n == 0:
        empty_idx = np.array([], dtype=np.int64)
        return empty_idx, embeddings, scores, {
            "enabled": False,
            "metric": metric,
            "drop_percent": float(drop_percent),
            "pool_size_before": 0,
            "pool_size_after": 0,
            "drop_count": 0,
            "keep_count": 0,
            "threshold": float("nan"),
            "score_stats": _scalar_stats(scores),
            "retained_score_stats": _scalar_stats(scores),
            stats_key: _scalar_stats(scores),
            retained_stats_key: _scalar_stats(scores),
        }

    if float(drop_percent) <= 0.0:
        local = np.arange(n, dtype=np.int64)
        return local, embeddings, scores, {
            "enabled": False,
            "metric": metric,
            "drop_percent": 0.0,
            "pool_size_before": n,
            "pool_size_after": n,
            "drop_count": 0,
            "keep_count": n,
            "threshold": float("-inf"),
            "score_stats": _scalar_stats(scores),
            "retained_score_stats": _scalar_stats(scores),
            stats_key: _scalar_stats(scores),
            retained_stats_key: _scalar_stats(scores),
        }

    keep_count = int(np.ceil((1.0 - float(drop_percent) / 100.0) * float(n)))
    keep_count = max(int(budget), keep_count)
    keep_count = max(1, min(keep_count, n))

    scores_np = scores.detach().cpu().numpy()
    # Sort by descending score, then ascending original position for deterministic ties.
    ranked = np.lexsort((np.arange(n, dtype=np.int64), -scores_np))
    retained_unsorted = ranked[:keep_count].astype(np.int64)
    retained_local = np.sort(retained_unsorted).astype(np.int64)
    retained_embeddings_t = torch.as_tensor(retained_local, dtype=torch.long, device=embeddings.device)
    retained_scores_t = torch.as_tensor(retained_local, dtype=torch.long, device=scores.device)
    filtered_embeddings = embeddings[retained_embeddings_t]
    retained_scores = scores[retained_scores_t]
    threshold = float(np.min(scores_np[retained_unsorted])) if keep_count > 0 else float("nan")
    return retained_local, filtered_embeddings, scores, {
        "enabled": True,
        "metric": metric,
        "drop_percent": float(drop_percent),
        "pool_size_before": n,
        "pool_size_after": int(keep_count),
        "drop_count": int(n - keep_count),
        "keep_count": int(keep_count),
        "threshold": threshold,
        "score_stats": _scalar_stats(scores),
        "retained_score_stats": _scalar_stats(retained_scores),
        stats_key: _scalar_stats(scores),
        retained_stats_key: _scalar_stats(retained_scores),
    }


def prefilter_by_secant_norm(
    embeddings: torch.Tensor,
    drop_percent: float,
    budget: int,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Drop bottom p% by D(x)=||phi(x)||_2."""
    d_scores = compute_secant_norm_scores(embeddings)
    candidate_local, filtered_embeddings, scores, debug = _prefilter_by_scores(
        embeddings=embeddings,
        scores=d_scores,
        drop_percent=drop_percent,
        budget=budget,
        metric="D",
    )
    debug["D_stats"] = debug["score_stats"]
    debug["D_retained_stats"] = debug["retained_score_stats"]
    return candidate_local, filtered_embeddings, scores, debug


def prefilter_by_clean_grad_norm(
    embeddings: torch.Tensor,
    g_clean: torch.Tensor,
    drop_percent: float,
    budget: int,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Drop bottom p% by D_clean(x)=||g_clean(x)||_2."""
    clean_scores = compute_clean_grad_norm_scores(g_clean)
    return prefilter_by_clean_grad_norm_scores(
        embeddings=embeddings,
        clean_grad_norm_scores=clean_scores,
        drop_percent=drop_percent,
        budget=budget,
    )


def prefilter_by_clean_grad_norm_scores(
    embeddings: torch.Tensor,
    clean_grad_norm_scores: torch.Tensor,
    drop_percent: float,
    budget: int,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Drop bottom p% by precomputed D_clean(x)=||g_clean(x)||_2 scores."""
    candidate_local, filtered_embeddings, scores, debug = _prefilter_by_scores(
        embeddings=embeddings,
        scores=clean_grad_norm_scores,
        drop_percent=drop_percent,
        budget=budget,
        metric="clean_grad_norm",
    )
    debug["clean_grad_norm_stats"] = debug["score_stats"]
    debug["clean_grad_norm_retained_stats"] = debug["retained_score_stats"]
    return candidate_local, filtered_embeddings, scores, debug


def prefilter_candidates_by_clean_grad_norm(
    indices: np.ndarray,
    g_clean: torch.Tensor,
    drop_percent: float,
    budget: int,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    """Return retained input indices after dropping the bottom p% by ||g_clean||."""
    if len(indices) != int(g_clean.size(0)):
        raise ValueError(f"indices length must match g_clean rows, got {len(indices)} vs {g_clean.size(0)}")
    scores = compute_clean_grad_norm_scores(g_clean)
    dummy_embeddings = g_clean.new_empty((g_clean.size(0), 0))
    retained_local, _, retained_scores, debug = _prefilter_by_scores(
        embeddings=dummy_embeddings,
        scores=scores,
        drop_percent=drop_percent,
        budget=budget,
        metric="clean_grad_norm",
    )
    return np.asarray(indices)[retained_local], retained_scores, debug


class OursSecantLogDetRefineStrategy(BaseAcquisition):
    method_name = "ours_secant_logdet_refine"

    def select(
        self,
        model,
        unlabeled_loader,
        labeled_loader,
        unlabeled_indices: np.ndarray,
        budget: int,
        device: torch.device,
        progress_logger=None,
    ) -> AcquisitionOutput:
        attack_type, attack_steps, attack_step_size, attack_random_start = _secant_attack_settings(self.cfg)

        t_score = time.perf_counter()
        embeddings, parts = compute_secant_badge_embeddings(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            epsilon_acq=float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            attack_type=attack_type,
            attack_steps=attack_steps,
            attack_step_size=attack_step_size,
            attack_random_start=attack_random_start,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            projection_dim=0,
            seed=int(self.cfg.seed),
            progress_logger=progress_logger,
            return_parts=True,
        )
        scoring_time = time.perf_counter() - t_score

        prefilter_metric = str(getattr(self.cfg, "prefilter_metric", "none")).lower()
        prefilter_drop_percent = float(getattr(self.cfg, "prefilter_drop_percent", 0.0))
        if prefilter_drop_percent > 0.0 and prefilter_metric in {"", "none", "off", "false"}:
            prefilter_metric = "d"
        if prefilter_metric in {"d", "secant_norm"}:
            candidate_local, embeddings_for_selection, d_scores, prefilter_debug = prefilter_by_secant_norm(
                embeddings=embeddings,
                drop_percent=prefilter_drop_percent,
                budget=budget,
            )
        elif prefilter_metric in {"clean_grad_norm", "clean_gradient_norm", "g_clean_norm", "d_clean"}:
            candidate_local, embeddings_for_selection, d_scores, prefilter_debug = prefilter_by_clean_grad_norm_scores(
                embeddings=embeddings,
                clean_grad_norm_scores=parts["clean_norm"],
                drop_percent=prefilter_drop_percent,
                budget=budget,
            )
        else:
            candidate_local, embeddings_for_selection, d_scores, prefilter_debug = prefilter_by_secant_norm(
                embeddings=embeddings,
                drop_percent=0.0,
                budget=budget,
            )

        lambda_reg = float(getattr(self.cfg, "logdet_lambda", getattr(self.cfg, "logdet_adv_disp_lambda", 1e-3)))
        score_chunk_size = int(getattr(self.cfg, "logdet_adv_disp_score_chunk_size", 8192))
        jitter = float(getattr(self.cfg, "logdet_adv_disp_jitter", 1e-8))
        swap_jitter = float(getattr(self.cfg, "logdet_adv_disp_swap_jitter", jitter))
        max_swap_rounds = int(getattr(self.cfg, "logdet_adv_disp_swap_max_rounds", 3))
        swap_top_unselected = int(getattr(self.cfg, "logdet_adv_disp_swap_top_unselected", 200))
        swap_top_selected = int(getattr(self.cfg, "logdet_adv_disp_swap_top_selected", 0))
        swap_improvement_tol = float(getattr(self.cfg, "logdet_adv_disp_swap_improvement_tol", 1e-8))

        t_select = time.perf_counter()
        # Selection is done in the dual Gram space on CPU/float64 for stability:
        # K is [N_pool,N_pool], while phi can be hundreds of thousands wide.
        kernel = build_gram_matrix(
            embeddings=embeddings_for_selection,
            chunk_size=max(1, min(score_chunk_size, 1024)),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        kernel_time = time.perf_counter() - t_select
        ambient_dim = int(embeddings_for_selection.size(1))

        picked_forward, forward_debug = forward_greedy_select(
            kernel=kernel,
            query_size=budget,
            lambda_reg=lambda_reg,
            score_chunk_size=score_chunk_size,
            jitter=jitter,
            ambient_dim=ambient_dim,
            progress_logger=progress_logger,
            progress_method_name="OURS_SECANT_LOGDET_REFINE_FORWARD",
        )
        forward_selection_time = time.perf_counter() - t_select - kernel_time

        t_refine = time.perf_counter()
        picked_refined, swap_debug = refine_by_swap(
            kernel=kernel,
            selected_local=picked_forward,
            lambda_reg=lambda_reg,
            score_chunk_size=score_chunk_size,
            jitter=swap_jitter,
            ambient_dim=ambient_dim,
            max_swap_rounds=max_swap_rounds,
            swap_top_unselected=swap_top_unselected,
            swap_top_selected=swap_top_selected,
            swap_improvement_tol=swap_improvement_tol,
            progress_logger=progress_logger,
            progress_method_name="OURS_SECANT_LOGDET_REFINE_SWAP",
        )
        refinement_time = time.perf_counter() - t_refine
        selection_time = time.perf_counter() - t_select

        picked_original_local = candidate_local[picked_refined]
        selected = unlabeled_indices[picked_original_local]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("ours_secant_logdet_refine selected duplicate unlabeled indices")
        if len(selected) != min(int(budget), int(len(unlabeled_indices))):
            raise ValueError(
                "ours_secant_logdet_refine returned unexpected number of samples: "
                f"{len(selected)} vs {min(int(budget), int(len(unlabeled_indices)))}"
            )

        clean_stats = tensor_stats(parts["clean_norm"])
        adv_stats = tensor_stats(parts["adv_norm"])
        correction_stats = tensor_stats(parts["correction_norm"])
        selected_score_mean = (
            float(d_scores[torch.as_tensor(picked_original_local, dtype=torch.long, device=d_scores.device)].mean().item())
            if len(picked_original_local) > 0
            else float("nan")
        )
        selected_clean_grad_norm_mean = (
            float(
                parts["clean_norm"][
                    torch.as_tensor(picked_original_local, dtype=torch.long, device=parts["clean_norm"].device)
                ]
                .mean()
                .item()
            )
            if len(picked_original_local) > 0
            else float("nan")
        )
        forward_obj = float(forward_debug.get("final_logdet_objective", float("nan")))
        refined_obj = float(swap_debug.get("final_logdet_objective", forward_obj))
        refinement_non_decrease = bool(refined_obj + max(jitter, swap_jitter) >= forward_obj)

        if progress_logger is not None:
            if bool(prefilter_debug["enabled"]):
                score_stats = prefilter_debug["score_stats"]
                progress_logger.log(
                    (
                        "[OURS_SECANT_LOGDET_REFINE_PREFILTER] "
                        f"metric={prefilter_debug['metric']} "
                        f"drop_percent={prefilter_debug['drop_percent']:.3f} "
                        f"pool={prefilter_debug['pool_size_before']}->{prefilter_debug['pool_size_after']} "
                        f"threshold={prefilter_debug['threshold']:.6f} "
                        f"score_min={score_stats['min']:.6f} score_median={score_stats['median']:.6f} "
                        f"score_max={score_stats['max']:.6f}"
                    ),
                    device=str(device),
                )
            progress_logger.log(
                (
                    "[OURS_SECANT_LOGDET_REFINE] "
                    f"selected={int(len(selected))} "
                    f"obj_forward={forward_obj:.6f} obj_refined={refined_obj:.6f} "
                    f"accepted_swaps={int(swap_debug.get('accepted_swaps', 0))} "
                    f"non_decrease={refinement_non_decrease}"
                ),
                device=str(device),
            )

        method_name = str(getattr(self.cfg, "acquisition_method", self.method_name)).lower()
        extras: Dict[str, Any] = {
            "method": method_name,
            "embedding": "concat_clean_last_layer_gradient_and_adv_minus_clean_gradient",
            "objective": "logdet_lambdaI_plus_sum_secant_phi_phiT",
            "selector_mode": "dual_gram_forward_greedy_plus_swap_refinement",
            "selection_linalg_device": "cpu",
            "selection_linalg_dtype": "float64",
            "kernel_build_dtype": "float32",
            "embedding_dim": int(embeddings.size(1)),
            "base_gradient_dim": int(parts["g_clean"].size(1)),
            "pool_size": int(len(unlabeled_indices)),
            "candidate_pool_size": int(embeddings_for_selection.size(0)),
            "selected_size": int(len(selected)),
            "selected_indices": selected.astype(np.int64).tolist(),
            "selected_local_indices": picked_original_local.astype(np.int64).tolist(),
            "lambda_reg": float(lambda_reg),
            "jitter": float(jitter),
            "score_chunk_size": int(score_chunk_size),
            "attack_type": attack_type,
            "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            "attack_steps": int(attack_steps),
            "attack_step_size": attack_step_size,
            "attack_random_start": bool(attack_random_start),
            "g_clean_norm_stats": clean_stats,
            "g_adv_norm_stats": adv_stats,
            "correction_norm_stats": correction_stats,
            "prefilter_enabled": bool(prefilter_debug["enabled"]),
            "prefilter_metric": prefilter_debug["metric"] if bool(prefilter_debug["enabled"]) else "none",
            "prefilter_drop_percent": float(prefilter_debug["drop_percent"]),
            "prefilter_pool_size_before": int(prefilter_debug["pool_size_before"]),
            "prefilter_pool_size_after": int(prefilter_debug["pool_size_after"]),
            "prefilter_drop_count": int(prefilter_debug["drop_count"]),
            "prefilter_keep_count": int(prefilter_debug["keep_count"]),
            "prefilter_threshold": float(prefilter_debug["threshold"]),
            "prefilter_score_stats": prefilter_debug["score_stats"],
            "prefilter_retained_score_stats": prefilter_debug["retained_score_stats"],
            "prefilter_selected_mean_score": selected_score_mean,
            "prefilter_threshold_D": float(prefilter_debug["threshold"])
            if prefilter_debug["metric"] == "D"
            else float("nan"),
            "prefilter_D_stats": prefilter_debug.get("D_stats", {}),
            "prefilter_retained_D_stats": prefilter_debug.get("D_retained_stats", {}),
            "prefilter_selected_mean_D": selected_score_mean
            if prefilter_debug["metric"] == "D"
            else float("nan"),
            "prefilter_threshold_clean_grad_norm": float(prefilter_debug["threshold"])
            if prefilter_debug["metric"] == "clean_grad_norm"
            else float("nan"),
            "prefilter_clean_grad_norm_stats": prefilter_debug.get("clean_grad_norm_stats", clean_stats),
            "prefilter_retained_clean_grad_norm_stats": prefilter_debug.get("clean_grad_norm_retained_stats", {}),
            "prefilter_selected_mean_clean_grad_norm": selected_clean_grad_norm_mean,
            "kernel_build_time_sec": float(kernel_time),
            "forward_selection_time_sec": float(forward_selection_time),
            "swap_refinement_time_sec": float(refinement_time),
            "forward_initial_logdet_objective": float(forward_debug.get("initial_logdet_objective", float("nan"))),
            "forward_final_logdet_objective": forward_obj,
            "forward_dual_final_logdet_objective": float(
                forward_debug.get("dual_final_logdet_objective", float("nan"))
            ),
            "forward_selected_log_marginal_gains": forward_debug.get("selected_log_marginal_gains", []),
            "forward_objectives": forward_debug.get("forward_objectives", []),
            "forward_nonfinite_score_steps": int(forward_debug.get("nonfinite_score_steps", 0)),
            "forward_inverse_rebuilds": int(forward_debug.get("inverse_rebuilds", 0)),
            "forward_used_pinv_rebuilds": int(forward_debug.get("used_pinv_rebuilds", 0)),
            "forward_max_jitter_used": float(forward_debug.get("max_jitter_used", 0.0)),
            "refinement_mode": "swap",
            "refinement_non_decrease": refinement_non_decrease,
            "refined_final_logdet_objective": refined_obj,
            "refinement_improvement": float(refined_obj - forward_obj),
            "swap_initial_logdet_objective": float(swap_debug.get("initial_logdet_objective", forward_obj)),
            "swap_final_logdet_objective": refined_obj,
            "swap_accepted_swaps": int(swap_debug.get("accepted_swaps", 0)),
            "swap_rounds_run": int(swap_debug.get("swap_rounds_run", 0)),
            "swap_best_gains_by_round": swap_debug.get("best_swap_gains_by_round", []),
            "swap_accepted_gains": swap_debug.get("accepted_swap_gains", []),
            "swap_state_rebuilds": int(swap_debug.get("state_rebuilds", 0)),
            "swap_state_used_pinv_rebuilds": int(swap_debug.get("state_used_pinv_rebuilds", 0)),
            "swap_nonfinite_swap_score_events": int(swap_debug.get("nonfinite_swap_score_events", 0)),
            "swap_max_jitter_used": float(swap_debug.get("max_jitter_used", 0.0)),
            "swap_top_unselected": int(swap_debug.get("swap_top_unselected", swap_top_unselected)),
            "swap_top_selected": int(swap_debug.get("swap_top_selected", swap_top_selected)),
            "swap_improvement_tol": float(swap_debug.get("swap_improvement_tol", swap_improvement_tol)),
        }

        return AcquisitionOutput(
            selected_indices=np.asarray(selected, dtype=np.int64),
            scores=None,
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras=extras,
        )

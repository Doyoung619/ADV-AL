import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from acquisition.badge import compute_badge_embeddings, select_badge_kmeanspp
from acquisition.badge_adv_mult import compute_adv_logit_mismatch_scores, normalize_adv_scores
from acquisition.utils import AcquisitionOutput, BaseAcquisition


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def compute_badge_embedding_norms(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    projection_dim: int = 0,
    seed: int = 0,
    progress_logger=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reuse standard BADGE embedding extraction and return:
      embeddings: g_BADGE(x)
      norms: ||g_BADGE(x)||_2
    """
    embeddings = compute_badge_embeddings(
        model=model,
        unlabeled_loader=unlabeled_loader,
        device=device,
        projection_dim=projection_dim,
        seed=seed,
        progress_logger=progress_logger,
    )
    norms = embeddings.float().norm(dim=1)
    return embeddings, norms


class BADGEAdvLagrangianStrategy(BaseAcquisition):
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
        total_t0 = time.perf_counter()

        t_badge = time.perf_counter()
        badge_embeddings, raw_badge_norms = compute_badge_embedding_norms(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            projection_dim=self.cfg.badge_projection_dim,
            seed=self.cfg.seed,
            progress_logger=progress_logger,
        )
        badge_time = time.perf_counter() - t_badge

        t_adv = time.perf_counter()
        raw_adv_scores = compute_adv_logit_mismatch_scores(
            model=model,
            unlabeled_loader=unlabeled_loader,
            epsilon_acq=self.cfg.epsilon_acq,
            attack_type=self.cfg.adv_attack_type_for_acquisition,
            pgd_steps=self.cfg.adv_pgd_steps,
            pgd_step_size=self.cfg.adv_pgd_step_size,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            device=device,
            progress_logger=progress_logger,
            progress_method_name="BADGE_ADV_LAGRANGIAN_ADV",
        )
        adv_time = time.perf_counter() - t_adv

        adv_norm_mode = self.cfg.score_normalization_adv or self.cfg.score_normalization
        badge_norm_mode = self.cfg.score_normalization_badge or self.cfg.score_normalization

        t_norm = time.perf_counter()
        norm_adv_scores = normalize_adv_scores(raw_adv_scores, mode=adv_norm_mode, tiny=self.cfg.tiny)
        norm_badge_scores = normalize_adv_scores(raw_badge_norms, mode=badge_norm_mode, tiny=self.cfg.tiny)
        lagrangian_scores = self.cfg.lambda_adv * norm_adv_scores + self.cfg.lambda_badge * norm_badge_scores
        norm_time = time.perf_counter() - t_norm

        t_filter = time.perf_counter()
        n_pool = int(lagrangian_scores.numel())
        candidate_target = max(int(budget), int(math.ceil(float(self.cfg.candidate_ratio) * float(max(1, budget)))))
        candidate_k = min(n_pool, candidate_target)
        if candidate_k > 0:
            candidate_local = torch.topk(lagrangian_scores, k=candidate_k, largest=True).indices.cpu().numpy()
        else:
            candidate_local = np.array([], dtype=np.int64)
        filter_time = time.perf_counter() - t_filter

        t_kmeans = time.perf_counter()
        if candidate_local.size > 0:
            candidate_embeddings = badge_embeddings[candidate_local]
            picked_in_candidates = select_badge_kmeanspp(candidate_embeddings, B=budget, seed=self.cfg.seed)
            picked_local = candidate_local[picked_in_candidates]
        else:
            picked_local = np.array([], dtype=np.int64)
        selected = unlabeled_indices[picked_local]
        kmeans_time = time.perf_counter() - t_kmeans

        raw_adv_stats = _tensor_stats(raw_adv_scores)
        raw_badge_stats = _tensor_stats(raw_badge_norms)
        norm_adv_stats = _tensor_stats(norm_adv_scores)
        norm_badge_stats = _tensor_stats(norm_badge_scores)
        lagrangian_stats = _tensor_stats(lagrangian_scores)

        total_time = time.perf_counter() - total_t0
        scoring_time = badge_time + adv_time + norm_time + filter_time
        selection_time = kmeans_time

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] raw_adv_stats "
                    f"min={raw_adv_stats['min']:.6f} max={raw_adv_stats['max']:.6f} "
                    f"mean={raw_adv_stats['mean']:.6f} std={raw_adv_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] raw_badge_norm_stats "
                    f"min={raw_badge_stats['min']:.6f} max={raw_badge_stats['max']:.6f} "
                    f"mean={raw_badge_stats['mean']:.6f} std={raw_badge_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] norm_adv_stats "
                    f"min={norm_adv_stats['min']:.6f} max={norm_adv_stats['max']:.6f} "
                    f"mean={norm_adv_stats['mean']:.6f} std={norm_adv_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] norm_badge_stats "
                    f"min={norm_badge_stats['min']:.6f} max={norm_badge_stats['max']:.6f} "
                    f"mean={norm_badge_stats['mean']:.6f} std={norm_badge_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] lagrangian_stats "
                    f"min={lagrangian_stats['min']:.6f} max={lagrangian_stats['max']:.6f} "
                    f"mean={lagrangian_stats['mean']:.6f} std={lagrangian_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] candidate_pool "
                    f"before={n_pool} after={int(candidate_local.size)}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_LAGRANGIAN] timing "
                    f"badge_embed={badge_time:.3f}s adv_scores={adv_time:.3f}s "
                    f"normalization={norm_time:.3f}s filtering={filter_time:.3f}s "
                    f"kmeanspp={kmeans_time:.3f}s total={total_time:.3f}s"
                ),
                device=str(device),
            )

        debug_data = None
        if self.cfg.debug_save_hybrid_scores:
            in_candidate = np.zeros(n_pool, dtype=np.int64)
            in_candidate[candidate_local] = 1
            is_selected = np.zeros(n_pool, dtype=np.int64)
            is_selected[picked_local] = 1
            debug_data = {
                "__file_tag": "hybrid_scores",
                "__column_order": [
                    "sample_index",
                    "raw_adv_score",
                    "raw_badge_norm",
                    "normalized_adv_score",
                    "normalized_badge_norm",
                    "lagrangian_score",
                    "in_candidate_subset",
                    "is_final_selected",
                ],
                "sample_index": np.asarray(unlabeled_indices, dtype=np.int64),
                "raw_adv_score": raw_adv_scores.numpy(),
                "raw_badge_norm": raw_badge_norms.numpy(),
                "normalized_adv_score": norm_adv_scores.numpy(),
                "normalized_badge_norm": norm_badge_scores.numpy(),
                "lagrangian_score": lagrangian_scores.numpy(),
                "in_candidate_subset": in_candidate,
                "is_final_selected": is_selected,
            }

        return AcquisitionOutput(
            selected_indices=selected,
            scores=lagrangian_scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "embedding_dim": int(badge_embeddings.size(1)),
                "projection_dim": int(self.cfg.badge_projection_dim),
                "epsilon_acq": float(self.cfg.epsilon_acq),
                "attack_type": self.cfg.adv_attack_type_for_acquisition,
                "pgd_steps": int(self.cfg.adv_pgd_steps),
                "pgd_step_size": self.cfg.adv_pgd_step_size,
                "lambda_adv": float(self.cfg.lambda_adv),
                "lambda_badge": float(self.cfg.lambda_badge),
                "candidate_ratio": float(self.cfg.candidate_ratio),
                "score_normalization": self.cfg.score_normalization,
                "score_normalization_adv": self.cfg.score_normalization_adv,
                "score_normalization_badge": self.cfg.score_normalization_badge,
                "tiny": float(self.cfg.tiny),
                "candidate_pool_size_before": int(n_pool),
                "candidate_pool_size_after": int(candidate_local.size),
                "raw_adv_stats": raw_adv_stats,
                "raw_badge_norm_stats": raw_badge_stats,
                "normalized_adv_stats": norm_adv_stats,
                "normalized_badge_stats": norm_badge_stats,
                "lagrangian_stats": lagrangian_stats,
                "timing_badge_embeddings_sec": float(badge_time),
                "timing_adv_scores_sec": float(adv_time),
                "timing_normalization_sec": float(norm_time),
                "timing_candidate_filtering_sec": float(filter_time),
                "timing_kmeanspp_sec": float(kmeans_time),
                "timing_total_acquisition_sec": float(total_time),
            },
            debug_data=debug_data,
        )

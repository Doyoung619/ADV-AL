import math
import time
from typing import Dict

import numpy as np
import torch

from acquisition.badge_adv_mult import compute_adv_logit_mismatch_scores, normalize_adv_scores
from acquisition.bald import score_bald
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


class BALDAdvLagrangianStrategy(BaseAcquisition):
    """
    Lagrangian BALD hybrid:
      s_lag(x) = lambda_adv * a_norm(x) + lambda_bald * bald_norm(x)
    Selection:
      1) keep top candidate_ratio * budget by s_lag
      2) final pick top-k by original BALD score inside candidates
    """

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

        t_bald = time.perf_counter()
        raw_bald_scores = score_bald(
            model=model,
            unlabeled_loader=unlabeled_loader,
            mc_passes=self.cfg.mc_passes,
            device=device,
            progress_logger=progress_logger,
        )
        bald_time = time.perf_counter() - t_bald

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
            progress_method_name="BALD_ADV_LAGRANGIAN_ADV",
        )
        adv_time = time.perf_counter() - t_adv

        t_norm = time.perf_counter()
        adv_norm_mode = self.cfg.score_normalization_adv or self.cfg.score_normalization
        bald_norm_mode = self.cfg.score_normalization_bald or self.cfg.score_normalization
        norm_adv_scores = normalize_adv_scores(raw_adv_scores, mode=adv_norm_mode, tiny=self.cfg.tiny)
        norm_bald_scores = normalize_adv_scores(raw_bald_scores, mode=bald_norm_mode, tiny=self.cfg.tiny)
        lagrangian_scores = self.cfg.lambda_adv * norm_adv_scores + self.cfg.lambda_bald * norm_bald_scores
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

        t_select = time.perf_counter()
        if candidate_local.size > 0:
            candidate_bald = raw_bald_scores[candidate_local]
            k = min(int(budget), int(candidate_bald.numel()))
            picked_in_candidates = torch.topk(candidate_bald, k=k, largest=True).indices.cpu().numpy()
            picked_local = candidate_local[picked_in_candidates]
        else:
            picked_local = np.array([], dtype=np.int64)
        selected = unlabeled_indices[picked_local]
        selection_time = time.perf_counter() - t_select

        raw_adv_stats = _tensor_stats(raw_adv_scores)
        raw_bald_stats = _tensor_stats(raw_bald_scores)
        norm_adv_stats = _tensor_stats(norm_adv_scores)
        norm_bald_stats = _tensor_stats(norm_bald_scores)
        lagrangian_stats = _tensor_stats(lagrangian_scores)
        total_time = time.perf_counter() - total_t0

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[BALD_ADV_LAGRANGIAN] raw_adv_stats "
                    f"min={raw_adv_stats['min']:.6f} max={raw_adv_stats['max']:.6f} "
                    f"mean={raw_adv_stats['mean']:.6f} std={raw_adv_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BALD_ADV_LAGRANGIAN] raw_bald_stats "
                    f"min={raw_bald_stats['min']:.6f} max={raw_bald_stats['max']:.6f} "
                    f"mean={raw_bald_stats['mean']:.6f} std={raw_bald_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BALD_ADV_LAGRANGIAN] candidate_pool "
                    f"before={n_pool} after={int(candidate_local.size)}"
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
                    "raw_bald_score",
                    "normalized_adv_score",
                    "normalized_bald_score",
                    "lagrangian_score",
                    "in_candidate_subset",
                    "is_final_selected",
                ],
                "sample_index": np.asarray(unlabeled_indices, dtype=np.int64),
                "raw_adv_score": raw_adv_scores.numpy(),
                "raw_bald_score": raw_bald_scores.numpy(),
                "normalized_adv_score": norm_adv_scores.numpy(),
                "normalized_bald_score": norm_bald_scores.numpy(),
                "lagrangian_score": lagrangian_scores.numpy(),
                "in_candidate_subset": in_candidate,
                "is_final_selected": is_selected,
            }

        return AcquisitionOutput(
            selected_indices=selected,
            scores=lagrangian_scores.numpy(),
            scoring_time_sec=float(bald_time + adv_time + norm_time + filter_time),
            selection_time_sec=float(selection_time),
            extras={
                "mc_passes": int(self.cfg.mc_passes),
                "epsilon_acq": float(self.cfg.epsilon_acq),
                "attack_type": self.cfg.adv_attack_type_for_acquisition,
                "pgd_steps": int(self.cfg.adv_pgd_steps),
                "pgd_step_size": self.cfg.adv_pgd_step_size,
                "lambda_adv": float(self.cfg.lambda_adv),
                "lambda_bald": float(self.cfg.lambda_bald),
                "candidate_ratio": float(self.cfg.candidate_ratio),
                "score_normalization": self.cfg.score_normalization,
                "score_normalization_adv": self.cfg.score_normalization_adv,
                "score_normalization_bald": self.cfg.score_normalization_bald,
                "tiny": float(self.cfg.tiny),
                "candidate_pool_size_before": int(n_pool),
                "candidate_pool_size_after": int(candidate_local.size),
                "raw_adv_stats": raw_adv_stats,
                "raw_bald_stats": raw_bald_stats,
                "normalized_adv_stats": norm_adv_stats,
                "normalized_bald_stats": norm_bald_stats,
                "lagrangian_stats": lagrangian_stats,
                "timing_bald_scores_sec": float(bald_time),
                "timing_adv_scores_sec": float(adv_time),
                "timing_normalization_sec": float(norm_time),
                "timing_candidate_filtering_sec": float(filter_time),
                "timing_selection_on_candidates_sec": float(selection_time),
                "timing_total_acquisition_sec": float(total_time),
            },
            debug_data=debug_data,
        )

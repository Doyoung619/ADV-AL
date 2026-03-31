import time
from typing import Dict

import numpy as np
import torch

from acquisition.entropy import score_entropy
from acquisition.utils import AcquisitionOutput, BaseAcquisition, compute_logit_mismatch_scores, tensor_stats


class EntropyDualBStrategy(BaseAcquisition):
    """
    entropy_dual_b:
      maximize c(x) subject to Entropy(x) >= kappa_b
    """

    @staticmethod
    def _selected_flag(n: int, picked_local: np.ndarray) -> np.ndarray:
        flag = np.zeros(n, dtype=np.int64)
        flag[picked_local] = 1
        return flag

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

        t_entropy = time.perf_counter()
        b_scores = score_entropy(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            use_mc=self.cfg.entropy_use_mc,
            mc_passes=self.cfg.mc_passes,
            progress_logger=progress_logger,
        ).float()
        entropy_time = time.perf_counter() - t_entropy

        t_c = time.perf_counter()
        c_scores = compute_logit_mismatch_scores(
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
            progress_method_name="ENTROPY_DUAL_B_C",
        ).float()
        c_time = time.perf_counter() - t_c

        t_filter = time.perf_counter()
        q = float(self.cfg.dual_percentile)
        kappa_b = float(torch.quantile(b_scores, q=q).item())
        feasible_mask = b_scores >= kappa_b
        feasible_local = torch.nonzero(feasible_mask, as_tuple=False).squeeze(1).cpu().numpy()
        filter_time = time.perf_counter() - t_filter

        t_select = time.perf_counter()
        feasible_c = c_scores[feasible_local]
        k = min(int(budget), int(feasible_c.numel()))
        picked_in_feasible = torch.topk(feasible_c, k=k, largest=True).indices.cpu().numpy()
        picked_local = feasible_local[picked_in_feasible]
        selected = unlabeled_indices[picked_local]
        selection_time = time.perf_counter() - t_select

        c_stats: Dict[str, float] = tensor_stats(c_scores)
        b_stats: Dict[str, float] = tensor_stats(b_scores)

        scoring_time = entropy_time + c_time + filter_time
        total_time = time.perf_counter() - total_t0

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[ENTROPY_DUAL_B] c_stats "
                    f"min={c_stats['min']:.6f} max={c_stats['max']:.6f} "
                    f"mean={c_stats['mean']:.6f} std={c_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[ENTROPY_DUAL_B] entropy_stats "
                    f"min={b_stats['min']:.6f} max={b_stats['max']:.6f} "
                    f"mean={b_stats['mean']:.6f} std={b_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[ENTROPY_DUAL_B] thresholds "
                    f"kappa_b(q={q:.3f})={kappa_b:.6f} feasible_size={int(feasible_local.size)} "
                    f"pool_size={int(len(unlabeled_indices))}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[ENTROPY_DUAL_B] timing "
                    f"entropy={entropy_time:.3f}s c_scores={c_time:.3f}s filter={filter_time:.3f}s "
                    f"selection={selection_time:.3f}s total={total_time:.3f}s"
                ),
                device=str(device),
            )

        selected_flag = self._selected_flag(n=len(unlabeled_indices), picked_local=picked_local)
        debug_data = {
            "__file_tag": "dual_scores",
            "__column_order": ["index", "c_score", "b_score", "feasible_flag", "selected_flag"],
            "index": np.asarray(unlabeled_indices, dtype=np.int64),
            "c_score": c_scores.numpy(),
            "b_score": b_scores.numpy(),
            "feasible_flag": feasible_mask.cpu().numpy().astype(np.int64),
            "selected_flag": selected_flag,
        }

        return AcquisitionOutput(
            selected_indices=selected,
            scores=c_scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": "entropy_dual_b",
                "entropy_use_mc": bool(self.cfg.entropy_use_mc),
                "mc_passes": int(self.cfg.mc_passes),
                "epsilon_acq": float(self.cfg.epsilon_acq),
                "attack_type": self.cfg.adv_attack_type_for_acquisition,
                "pgd_steps": int(self.cfg.adv_pgd_steps),
                "pgd_step_size": self.cfg.adv_pgd_step_size,
                "kappa_b_quantile": float(q),
                "kappa_b": float(kappa_b),
                "feasible_size": int(feasible_local.size),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
                "c_stats": c_stats,
                "b_stats": b_stats,
                "timing_entropy_scores_sec": float(entropy_time),
                "timing_c_scores_sec": float(c_time),
                "timing_filter_sec": float(filter_time),
                "timing_selection_sec": float(selection_time),
                "timing_total_acquisition_sec": float(total_time),
            },
            debug_data=debug_data,
        )

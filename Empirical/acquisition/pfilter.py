import time
from typing import Callable

import numpy as np
import torch

from acquisition.bald import score_bald
from acquisition.entropy import score_entropy
from acquisition.utils import AcquisitionOutput, BaseAcquisition, choose_without_replacement, topk_unlabeled_indices


def _select_from_percentile_feasible(
    base_scores: torch.Tensor,
    unlabeled_indices: np.ndarray,
    budget: int,
    percentile: float,
    selector: Callable[[torch.Tensor, np.ndarray, int, int], np.ndarray],
    seed: int,
) -> np.ndarray:
    q = float(percentile)
    threshold = float(torch.quantile(base_scores, q=q).item())
    feasible_mask = base_scores >= threshold
    feasible_local = torch.nonzero(feasible_mask, as_tuple=False).squeeze(1).cpu().numpy()
    if feasible_local.size == 0:
        return np.array([], dtype=np.int64)
    feasible_scores = base_scores[feasible_local]
    chosen_local = selector(feasible_scores, feasible_local, budget, seed)
    return unlabeled_indices[chosen_local]


def _random_selector(feasible_scores: torch.Tensor, feasible_local: np.ndarray, budget: int, seed: int) -> np.ndarray:
    _ = feasible_scores
    return choose_without_replacement(feasible_local, budget=budget, seed=seed)


def _topk_selector(feasible_scores: torch.Tensor, feasible_local: np.ndarray, budget: int, seed: int) -> np.ndarray:
    _ = seed
    k = min(int(budget), int(feasible_scores.numel()))
    picked_in_feasible = torch.topk(feasible_scores, k=k, largest=True).indices.cpu().numpy()
    return feasible_local[picked_in_feasible]


class EntropyPercentileRandomStrategy(BaseAcquisition):
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
        _ = labeled_loader
        t0 = time.perf_counter()
        scores = score_entropy(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            use_mc=self.cfg.entropy_use_mc,
            mc_passes=self.cfg.mc_passes,
            progress_logger=progress_logger,
        ).float()
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = _select_from_percentile_feasible(
            base_scores=scores,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            percentile=self.cfg.dual_percentile,
            selector=_random_selector,
            seed=self.cfg.seed,
        )
        selection_time = time.perf_counter() - t1
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={"dual_percentile": float(self.cfg.dual_percentile), "base_score": "entropy"},
        )


class EntropyPercentileEntropyStrategy(BaseAcquisition):
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
        _ = labeled_loader
        t0 = time.perf_counter()
        scores = score_entropy(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            use_mc=self.cfg.entropy_use_mc,
            mc_passes=self.cfg.mc_passes,
            progress_logger=progress_logger,
        ).float()
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = _select_from_percentile_feasible(
            base_scores=scores,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            percentile=self.cfg.dual_percentile,
            selector=_topk_selector,
            seed=self.cfg.seed,
        )
        selection_time = time.perf_counter() - t1
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={"dual_percentile": float(self.cfg.dual_percentile), "base_score": "entropy"},
        )


class BALDPercentileRandomStrategy(BaseAcquisition):
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
        _ = labeled_loader
        t0 = time.perf_counter()
        scores = score_bald(
            model=model,
            unlabeled_loader=unlabeled_loader,
            mc_passes=self.cfg.mc_passes,
            device=device,
            progress_logger=progress_logger,
        ).float()
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = _select_from_percentile_feasible(
            base_scores=scores,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            percentile=self.cfg.dual_percentile,
            selector=_random_selector,
            seed=self.cfg.seed,
        )
        selection_time = time.perf_counter() - t1
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={"dual_percentile": float(self.cfg.dual_percentile), "base_score": "bald"},
        )


class BALDPercentileBALDStrategy(BaseAcquisition):
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
        _ = labeled_loader
        t0 = time.perf_counter()
        scores = score_bald(
            model=model,
            unlabeled_loader=unlabeled_loader,
            mc_passes=self.cfg.mc_passes,
            device=device,
            progress_logger=progress_logger,
        ).float()
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = _select_from_percentile_feasible(
            base_scores=scores,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            percentile=self.cfg.dual_percentile,
            selector=_topk_selector,
            seed=self.cfg.seed,
        )
        selection_time = time.perf_counter() - t1
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={"dual_percentile": float(self.cfg.dual_percentile), "base_score": "bald"},
        )

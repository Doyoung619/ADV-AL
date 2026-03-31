import numpy as np
import torch
import time

from acquisition.utils import AcquisitionOutput, BaseAcquisition, choose_without_replacement


class RandomStrategy(BaseAcquisition):
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
        t0 = time.perf_counter()
        selected = choose_without_replacement(unlabeled_indices, budget=budget, seed=self.cfg.seed)
        selection_time = time.perf_counter() - t0
        return AcquisitionOutput(
            selected_indices=selected,
            scores=None,
            scoring_time_sec=0.0,
            selection_time_sec=selection_time,
            extras={},
        )

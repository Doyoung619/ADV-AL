import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition, topk_unlabeled_indices


def score_bald(
    model,
    unlabeled_loader,
    mc_passes: int = 20,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
    progress_logger=None,
):
    if device is None:
        device = next(model.parameters()).device

    prev_training = model.training
    model.train()  # required for MC-dropout at acquisition time
    all_scores = []

    with torch.no_grad():
        total_batches = len(unlabeled_loader)
        t0 = time.perf_counter()
        for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
            images = images.to(device, non_blocking=True)
            probs_t = []
            for _ in range(mc_passes):
                # Runtime bottleneck: MC-dropout repeats model forward pass T times per batch.
                logits = model(images)
                probs_t.append(F.softmax(logits, dim=1))

            probs = torch.stack(probs_t, dim=0)  # [T, B, C]
            p_bar = probs.mean(dim=0)  # [B, C]

            predictive_entropy = -(p_bar * torch.log(p_bar + eps)).sum(dim=1)  # [B]
            expected_entropy = (-(probs * torch.log(probs + eps)).sum(dim=2)).mean(dim=0)  # [B]
            score = predictive_entropy - expected_entropy
            all_scores.append(score.cpu())

            if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
                progress_logger.log_scoring_eta(
                    method="BALD",
                    processed_batches=batch_idx,
                    total_batches=total_batches,
                    elapsed=time.perf_counter() - t0,
                    device=str(device),
                )

    model.train(prev_training)
    return torch.cat(all_scores, dim=0)


class BALDStrategy(BaseAcquisition):
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
        scores = score_bald(
            model,
            unlabeled_loader,
            mc_passes=self.cfg.mc_passes,
            device=device,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = topk_unlabeled_indices(unlabeled_indices, scores, budget)
        selection_time = time.perf_counter() - t1
        return AcquisitionOutput(
            selected_indices=selected,
            scores=scores.numpy(),
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras={"mc_passes": self.cfg.mc_passes},
        )

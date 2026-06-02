import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition, tensor_stats, topk_unlabeled_indices


@torch.no_grad()
def score_margin(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    progress_logger=None,
) -> torch.Tensor:
    """
    Margin sampling score per sample.

    margin(x) = p1(x) - p2(x), where p1 >= p2 are top-2 class probabilities.
    We return score(x) = -margin(x) so that higher score means higher priority
    when using top-k selection utilities that pick largest scores.
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    all_scores = []

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
        scores = -margins
        all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="Margin",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    if was_training:
        model.train()

    return torch.cat(all_scores, dim=0)


class MarginStrategy(BaseAcquisition):
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
        scores = score_margin(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            progress_logger=progress_logger,
        ).float()
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        selected = topk_unlabeled_indices(unlabeled_indices, scores, budget)
        selection_time = time.perf_counter() - t1

        score_stats = tensor_stats(scores)
        if progress_logger is not None:
            progress_logger.log(
                (
                    "[MARGIN] score_stats "
                    f"min={score_stats['min']:.6f} max={score_stats['max']:.6f} "
                    f"mean={score_stats['mean']:.6f} std={score_stats['std']:.6f}"
                ),
                device=str(device),
            )

        return AcquisitionOutput(
            selected_indices=selected,
            scores=scores.numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": "margin",
                "score_definition": "score=-margin where margin=top1_prob-top2_prob",
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "score_stats": score_stats,
            },
        )


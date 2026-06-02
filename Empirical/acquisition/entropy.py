import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition, topk_unlabeled_indices
from models import enable_dropout_layers_only


@torch.no_grad()
def score_entropy(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
    use_mc: bool = False,
    mc_passes: int = 20,
    progress_logger=None,
):
    if device is None:
        device = next(model.parameters()).device

    all_scores = []
    if use_mc:
        enable_dropout_layers_only(model)
    else:
        model.eval()

    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        if not use_mc:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
        else:
            probs_mc = []
            for _ in range(mc_passes):
                logits = model(images)
                probs_mc.append(F.softmax(logits, dim=1))
            probs = torch.stack(probs_mc, dim=0).mean(dim=0)

        entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
        all_scores.append(entropy.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="Entropy",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


class EntropyStrategy(BaseAcquisition):
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
        scores = score_entropy(
            model,
            unlabeled_loader,
            device=device,
            use_mc=self.cfg.entropy_use_mc,
            mc_passes=self.cfg.mc_passes,
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
            extras={},
        )

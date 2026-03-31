import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition


@torch.no_grad()
def compute_badge_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    projection_dim: int = 0,
    seed: int = 0,
    progress_logger=None,
):
    """
    Compute BADGE gradient embeddings on the final linear layer.

    If h(x) has shape [D], p(x) has shape [C], and y_hat=argmax_c p_c:
      g_x = concat_c ((p_c - 1[c=y_hat]) * h(x))
    Output embedding shape: [N_pool, C*D] (or [N_pool, projection_dim] if projected).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    embeddings = []
    proj = None
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)  # logits [B,C], features [B,D]
        probs = F.softmax(logits, dim=1)  # [B,C]
        y_hat = probs.argmax(dim=1)  # [B]

        one_hot = F.one_hot(y_hat, num_classes=probs.size(1)).float()  # [B,C]
        coeff = probs - one_hot  # [B,C]

        # [B, C, D]: per-class last-layer gradient block
        grad_embed = coeff.unsqueeze(2) * features.unsqueeze(1)
        grad_embed = grad_embed.reshape(grad_embed.size(0), -1)  # [B, C*D]

        if projection_dim > 0 and projection_dim < grad_embed.size(1):
            if proj is None:
                g = torch.Generator(device=grad_embed.device)
                g.manual_seed(seed)
                proj = torch.randn(
                    grad_embed.size(1),
                    projection_dim,
                    generator=g,
                    device=grad_embed.device,
                    dtype=grad_embed.dtype,
                ) / math.sqrt(float(projection_dim))
            grad_embed = grad_embed @ proj

        embeddings.append(grad_embed.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="BADGE",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(embeddings, dim=0)


def select_badge_kmeanspp(embeddings: torch.Tensor, B: int, seed: int) -> np.ndarray:
    """
    K-means++ seeding on BADGE embeddings.
    Returns indices of selected examples (seeded points, not centroids).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N, D]")

    N = embeddings.size(0)
    B = min(B, N)
    if B <= 0:
        return np.array([], dtype=np.int64)

    x = embeddings.float()
    norms = (x * x).sum(dim=1)  # [N]
    rng = np.random.default_rng(seed)

    selected = []
    first = int(rng.integers(0, N))
    selected.append(first)

    center = x[first : first + 1]
    min_dist_sq = (norms + norms[first] - 2.0 * (x @ center.t()).squeeze(1)).clamp_min(0.0)
    min_dist_sq[first] = 0.0

    for _ in range(1, B):
        # Runtime bottleneck: repeated full-pool distance updates are O(N * B * D).
        probs = min_dist_sq.cpu().numpy()
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # Degenerate case: choose uniformly among not-yet-selected points.
            remaining = np.setdiff1d(np.arange(N), np.array(selected), assume_unique=False)
            if len(remaining) == 0:
                break
            next_idx = int(rng.choice(remaining))
        else:
            probs = probs / probs_sum
            next_idx = int(rng.choice(N, p=probs))
            while next_idx in selected:
                next_idx = int(rng.choice(N, p=probs))

        selected.append(next_idx)
        c = x[next_idx : next_idx + 1]
        dist_sq = (norms + norms[next_idx] - 2.0 * (x @ c.t()).squeeze(1)).clamp_min(0.0)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        min_dist_sq[next_idx] = 0.0

    return np.array(selected, dtype=np.int64)


class BADGEStrategy(BaseAcquisition):
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
        embeddings = compute_badge_embeddings(
            model,
            unlabeled_loader,
            device=device,
            projection_dim=self.cfg.badge_projection_dim,
            seed=self.cfg.seed,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t0

        candidate_local = np.arange(embeddings.size(0), dtype=np.int64)
        if self.cfg.badge_candidate_cap is not None and self.cfg.badge_candidate_cap < embeddings.size(0):
            rng = np.random.default_rng(self.cfg.seed)
            candidate_local = rng.choice(candidate_local, size=self.cfg.badge_candidate_cap, replace=False)
            embeddings_sel = embeddings[candidate_local]
        else:
            embeddings_sel = embeddings

        t1 = time.perf_counter()
        picked_local_in_candidates = select_badge_kmeanspp(embeddings_sel, B=budget, seed=self.cfg.seed)
        picked_local = candidate_local[picked_local_in_candidates]
        selected = unlabeled_indices[picked_local]
        selection_time = time.perf_counter() - t1

        return AcquisitionOutput(
            selected_indices=selected,
            scores=None,
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras={
                "embedding_dim": int(embeddings.size(1)),
                "projection_dim": int(self.cfg.badge_projection_dim),
                "candidate_cap": self.cfg.badge_candidate_cap,
            },
        )

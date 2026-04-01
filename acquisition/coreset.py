import time
from typing import Optional, Tuple

import numpy as np
import torch

from acquisition.utils import AcquisitionOutput, BaseAcquisition


@torch.no_grad()
def extract_penultimate_features(
    model,
    loader,
    device: Optional[torch.device] = None,
    progress_logger=None,
    tag: str = "CoreSet",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract penultimate features (input to final classifier layer).
    Assumes model supports forward(..., return_features=True).
    """
    if device is None:
        device = next(model.parameters()).device

    prev_training = model.training
    model.eval()
    feats = []
    indices = []
    total_batches = len(loader)
    t0 = time.perf_counter()

    try:
        for batch_idx, (images, _, batch_indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits, features = model(images, return_features=True)
            _ = logits
            feats.append(features.cpu())
            indices.append(batch_indices.clone())

            if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
                progress_logger.log_scoring_eta(
                    method=f"{tag}-feat",
                    processed_batches=batch_idx,
                    total_batches=total_batches,
                    elapsed=time.perf_counter() - t0,
                    device=str(device),
                )
    finally:
        model.train(prev_training)

    if len(feats) == 0:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
    return torch.cat(feats, dim=0), torch.cat(indices, dim=0)


def _min_l2_distance_to_centers(
    points: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Compute min_j ||points_i - centers_j||_2 for all i.
    """
    n = points.size(0)
    if n == 0:
        return torch.empty((0,), dtype=points.dtype, device=points.device)
    if centers.size(0) == 0:
        return torch.full((n,), float("inf"), dtype=points.dtype, device=points.device)

    out = torch.empty((n,), dtype=points.dtype, device=points.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = points[start:end]
        d = torch.cdist(chunk, centers, p=2.0)
        out[start:end] = d.min(dim=1).values
    return out


def kcenter_greedy(
    unlabeled_features: torch.Tensor,
    labeled_features: Optional[torch.Tensor],
    budget: int,
    chunk_size: int = 2048,
    progress_logger=None,
) -> Tuple[np.ndarray, dict]:
    """
    Greedy k-Center selection initialized with labeled centers S0.
    """
    if unlabeled_features.ndim != 2:
        raise ValueError(f"unlabeled_features must be rank-2 [N,D], got shape={tuple(unlabeled_features.shape)}")

    points = unlabeled_features.to(dtype=torch.float32, device=torch.device("cpu"))
    n = points.size(0)
    if n == 0 or budget <= 0:
        return np.array([], dtype=np.int64), {"cover_radius_history": []}

    k = min(int(budget), int(n))
    selected = []
    available = torch.ones(n, dtype=torch.bool, device=torch.device("cpu"))

    if labeled_features is not None and labeled_features.numel() > 0:
        centers0 = labeled_features.to(dtype=torch.float32, device=torch.device("cpu"))
        min_dist = _min_l2_distance_to_centers(points=points, centers=centers0, chunk_size=chunk_size)
    else:
        # Robust fallback when S0 is empty: start from farthest-from-mean point.
        center_mean = points.mean(dim=0, keepdim=True)
        init_idx = int(torch.argmax(torch.cdist(points, center_mean, p=2.0).squeeze(1)).item())
        selected.append(init_idx)
        available[init_idx] = False
        min_dist = _min_l2_distance_to_centers(points=points, centers=points[init_idx : init_idx + 1], chunk_size=chunk_size)
        min_dist[~available] = -1.0

    cover_radius_history = []

    while len(selected) < k:
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        available[idx] = False

        new_center = points[idx : idx + 1]
        d_new = _min_l2_distance_to_centers(points=points, centers=new_center, chunk_size=chunk_size)
        min_dist = torch.minimum(min_dist, d_new)
        min_dist[~available] = -1.0

        if available.any():
            cover_radius = float(min_dist[available].max().item())
        else:
            cover_radius = 0.0
        cover_radius_history.append(cover_radius)

        if progress_logger is not None and ((len(selected) % 10 == 0) or (len(selected) == k)):
            progress_logger.log(f"[CoreSet] greedy {len(selected)}/{k} cover_radius={cover_radius:.6f}")

    debug = {
        "cover_radius_history": cover_radius_history,
        "distance_metric": "l2_penultimate",
        "chunk_size": int(chunk_size),
        "num_unlabeled": int(n),
        "num_labeled_centers": int(0 if labeled_features is None else labeled_features.size(0)),
    }
    return np.asarray(selected, dtype=np.int64), debug


class CoreSetStrategy(BaseAcquisition):
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
        t_score = time.perf_counter()
        unlabeled_features, _ = extract_penultimate_features(
            model=model,
            loader=unlabeled_loader,
            device=device,
            progress_logger=progress_logger,
            tag="CoreSet-unlabeled",
        )
        if labeled_loader is not None:
            labeled_features, _ = extract_penultimate_features(
                model=model,
                loader=labeled_loader,
                device=device,
                progress_logger=progress_logger,
                tag="CoreSet-labeled",
            )
        else:
            labeled_features = torch.empty((0, unlabeled_features.size(1)), dtype=unlabeled_features.dtype)
        scoring_time = time.perf_counter() - t_score

        t_sel = time.perf_counter()
        picked_local, debug = kcenter_greedy(
            unlabeled_features=unlabeled_features,
            labeled_features=labeled_features,
            budget=budget,
            chunk_size=2048,
            progress_logger=progress_logger,
        )
        selection_time = time.perf_counter() - t_sel
        selected = unlabeled_indices[picked_local]

        return AcquisitionOutput(
            selected_indices=selected,
            scores=None,
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras=debug,
        )

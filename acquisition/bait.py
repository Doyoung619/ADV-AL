import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition


@torch.no_grad()
def compute_last_layer_features_and_probs(model, loader, device: Optional[torch.device] = None, progress_logger=None, tag: str = "BAIT"):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    feats, probs, indices = [], [], []
    total_batches = len(loader)
    t0 = time.perf_counter()
    for batch_idx, (images, _, batch_indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        logits, h = model(images, return_features=True)  # h: [B, D]
        p = F.softmax(logits, dim=1)  # [B, C]
        feats.append(h.cpu())
        probs.append(p.cpu())
        indices.append(batch_indices.clone())
        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=f"{tag}-feat",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )
    return torch.cat(feats, dim=0), torch.cat(probs, dim=0), torch.cat(indices, dim=0)


def compute_pointwise_fisher_factor(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Construct V_x such that I(x) = V_x V_x^T for:
      I(x) = h h^T kron (diag(p) - p p^T)

    Args:
      h: [D]
      p: [C]
    Returns:
      V_x: [C*D, r], r <= C (rank from eig factor of class covariance)
    """
    h = h.view(-1)
    p = p.view(-1)
    c = p.numel()
    d = h.numel()

    s = torch.diag(p) - torch.outer(p, p)  # [C, C]
    eigvals, eigvecs = torch.linalg.eigh(s)
    keep = eigvals > 1e-10
    if keep.sum() == 0:
        return torch.zeros((c * d, 1), dtype=h.dtype, device=h.device)

    L = eigvecs[:, keep] * torch.sqrt(eigvals[keep]).unsqueeze(0)  # [C, r]
    # kron(L[:, r], h) for each rank component r:
    # output shape [C*D, r]
    v = torch.einsum("cr,d->cdr", L, h).reshape(c * d, -1)
    return v


def _make_projection(in_dim: int, projection_dim: int, seed: int, dtype: torch.dtype, device: torch.device):
    if projection_dim <= 0 or projection_dim >= in_dim:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Gaussian random projection with 1/sqrt(m) scaling.
    proj = torch.randn(projection_dim, in_dim, generator=gen, dtype=dtype, device=device) / math.sqrt(float(projection_dim))
    return proj


def _factor_projected(
    h: torch.Tensor,
    p: torch.Tensor,
    projection: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    factor = compute_pointwise_fisher_factor(h.to(device=device, dtype=dtype), p.to(device=device, dtype=dtype))
    if projection is not None:
        factor = projection @ factor
    return factor


def _average_fisher(
    features: torch.Tensor,
    probs: torch.Tensor,
    projection: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    progress_logger=None,
    tag: str = "fisher",
) -> torch.Tensor:
    dim = projection.size(0) if projection is not None else features.size(1) * probs.size(1)
    fisher = torch.zeros((dim, dim), dtype=dtype, device=device)
    n = features.size(0)
    start = time.perf_counter()
    for i in range(n):
        # Runtime bottleneck: per-sample Fisher accumulation dominates BAIT scoring time.
        v = _factor_projected(features[i], probs[i], projection, dtype=dtype, device=device)  # [dim, r]
        fisher += v @ v.t()
        if progress_logger is not None and ((i + 1) % 500 == 0 or i + 1 == n):
            elapsed = time.perf_counter() - start
            avg = elapsed / float(i + 1)
            eta = avg * (n - (i + 1))
            progress_logger.log(
                f"[BAIT] {tag} {i + 1}/{n} elapsed={elapsed:.1f}s avg/item={avg:.4f}s eta={eta:.1f}s"
            )
    fisher /= max(1, n)
    return fisher


def _stable_inverse(matrix: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    try:
        return torch.linalg.inv(matrix)
    except RuntimeError:
        return torch.linalg.pinv(matrix + jitter * eye)


def select_bait(
    features: torch.Tensor,
    probs: torch.Tensor,
    labeled_features: torch.Tensor,
    labeled_probs: torch.Tensor,
    B: int,
    lambda_reg: float,
    candidate_cap: Optional[int] = None,
    projection_dim: int = 256,
    seed: int = 0,
    dtype: str = "float32",
    progress_logger=None,
):
    """
    BAIT forward-backward selector in projected last-layer Fisher space.

    Forward:
      pick 2B samples minimizing tr((M + I(x))^{-1} I_pool)
    Backward:
      remove B samples minimizing tr((M - I(x))^{-1} I_pool)
    """
    if features.size(0) == 0:
        return np.array([], dtype=np.int64), {"forward_objectives": [], "backward_objectives": []}

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    device = torch.device("cpu")

    n_unlabeled, feat_dim = features.shape
    num_classes = probs.shape[1]
    full_dim = feat_dim * num_classes
    projection = _make_projection(full_dim, projection_dim=projection_dim, seed=seed, dtype=torch_dtype, device=device)
    work_dim = projection.size(0) if projection is not None else full_dim

    rng = np.random.default_rng(seed)
    factor_cache: Dict[int, torch.Tensor] = {}

    def get_unlabeled_factor(idx: int) -> torch.Tensor:
        if idx not in factor_cache:
            factor_cache[idx] = _factor_projected(features[idx], probs[idx], projection, dtype=torch_dtype, device=device)
        return factor_cache[idx]

    t_pool = time.perf_counter()
    I_pool = _average_fisher(
        features=features,
        probs=probs,
        projection=projection,
        dtype=torch_dtype,
        device=device,
        progress_logger=progress_logger,
        tag="I_pool",
    )
    pool_time = time.perf_counter() - t_pool

    I_labeled = _average_fisher(
        features=labeled_features,
        probs=labeled_probs,
        projection=projection,
        dtype=torch_dtype,
        device=device,
        progress_logger=progress_logger,
        tag="I_labeled",
    )

    M0 = I_labeled + lambda_reg * torch.eye(work_dim, dtype=torch_dtype, device=device)
    M_inv = _stable_inverse(M0)

    oversample = min(2 * B, n_unlabeled)
    remaining_mask = np.ones(n_unlabeled, dtype=bool)
    selected = []
    forward_objectives = []

    for step in range(oversample):
        available = np.where(remaining_mask)[0]
        if len(available) == 0:
            break
        if candidate_cap is not None and len(available) > candidate_cap:
            candidates = rng.choice(available, size=candidate_cap, replace=False)
        else:
            candidates = available

        base_trace = torch.trace(M_inv @ I_pool).item()
        best_obj = float("inf")
        best_idx = None

        for idx in candidates:
            # Runtime bottleneck: repeated candidate objective evaluation in greedy forward pass.
            v = get_unlabeled_factor(int(idx))  # [work_dim, r]
            u = M_inv @ v  # [work_dim, r]
            mid = torch.eye(v.size(1), dtype=torch_dtype, device=device) + v.t() @ u
            try:
                mid_inv = torch.linalg.inv(mid)
            except RuntimeError:
                mid_inv = torch.linalg.pinv(mid)
            iu = I_pool @ u
            reduction = torch.trace(mid_inv @ (u.t() @ iu)).item()
            obj = base_trace - reduction
            if obj < best_obj:
                best_obj = obj
                best_idx = int(idx)

        if best_idx is None:
            break

        v = get_unlabeled_factor(best_idx)
        u = M_inv @ v
        mid = torch.eye(v.size(1), dtype=torch_dtype, device=device) + v.t() @ u
        mid_inv = _stable_inverse(mid)
        M_inv = M_inv - u @ mid_inv @ u.t()

        selected.append(best_idx)
        remaining_mask[best_idx] = False
        forward_objectives.append(best_obj)

        if progress_logger is not None and ((step + 1) % 20 == 0 or step + 1 == oversample):
            progress_logger.log(
                f"[BAIT] forward {step + 1}/{oversample} best_obj={best_obj:.6f} cache={len(factor_cache)}"
            )

    current = selected.copy()
    backward_objectives = []

    while len(current) > B:
        base_trace = torch.trace(M_inv @ I_pool).item()
        best_remove_obj = float("inf")
        best_remove_idx = None
        best_terms = None

        for idx in current:
            # Runtime bottleneck: repeated downdate objective checks in backward pruning.
            v = get_unlabeled_factor(int(idx))
            u = M_inv @ v
            mid = torch.eye(v.size(1), dtype=torch_dtype, device=device) - v.t() @ u
            try:
                mid_inv = torch.linalg.inv(mid)
            except RuntimeError:
                continue

            iu = I_pool @ u
            increase = torch.trace(mid_inv @ (u.t() @ iu)).item()
            obj = base_trace + increase
            if obj < best_remove_obj:
                best_remove_obj = obj
                best_remove_idx = int(idx)
                best_terms = (u, mid_inv)

        if best_remove_idx is None:
            # Fall back to random drop in numerically degenerate cases.
            best_remove_idx = int(current[-1])
            v = get_unlabeled_factor(best_remove_idx)
            u = M_inv @ v
            mid = torch.eye(v.size(1), dtype=torch_dtype, device=device) - v.t() @ u
            mid_inv = _stable_inverse(mid)
            best_terms = (u, mid_inv)

        u, mid_inv = best_terms
        M_inv = M_inv + u @ mid_inv @ u.t()

        current.remove(best_remove_idx)
        backward_objectives.append(best_remove_obj)

        if progress_logger is not None and (len(current) % 20 == 0 or len(current) == B):
            progress_logger.log(f"[BAIT] backward keep={len(current)}/{B} best_obj={best_remove_obj:.6f}")

    debug = {
        "forward_objectives": forward_objectives,
        "backward_objectives": backward_objectives,
        "pool_fisher_time_sec": pool_time,
        "projection_dim": int(work_dim),
        "full_dim": int(full_dim),
        "candidate_cap": candidate_cap,
    }
    return np.asarray(current, dtype=np.int64), debug


class BAITStrategy(BaseAcquisition):
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
        unlabeled_features, unlabeled_probs, _ = compute_last_layer_features_and_probs(
            model, unlabeled_loader, device=device, progress_logger=progress_logger, tag="BAIT-unlabeled"
        )
        labeled_features, labeled_probs, _ = compute_last_layer_features_and_probs(
            model, labeled_loader, device=device, progress_logger=progress_logger, tag="BAIT-labeled"
        )
        scoring_time = time.perf_counter() - t_score

        t_sel = time.perf_counter()
        picked_local, debug = select_bait(
            features=unlabeled_features,
            probs=unlabeled_probs,
            labeled_features=labeled_features,
            labeled_probs=labeled_probs,
            B=budget,
            lambda_reg=self.cfg.lambda_reg,
            candidate_cap=self.cfg.candidate_cap,
            projection_dim=self.cfg.bait_projection_dim,
            seed=self.cfg.seed,
            dtype=self.cfg.bait_dtype,
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

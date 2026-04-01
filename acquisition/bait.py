import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.t())


def _inverse_spd_with_jitter(
    matrix: torch.Tensor,
    base_jitter: float = 1e-12,
    max_tries: int = 7,
) -> torch.Tensor:
    maybe_inv = _try_inverse_spd_with_jitter(matrix=matrix, base_jitter=base_jitter, max_tries=max_tries)
    if maybe_inv is not None:
        return maybe_inv
    matrix = _symmetrize(matrix)
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    jitter = base_jitter * (10.0 ** max(max_tries - 1, 0))
    return torch.linalg.pinv(matrix + jitter * eye)


def _try_inverse_spd_with_jitter(
    matrix: torch.Tensor,
    base_jitter: float = 1e-12,
    max_tries: int = 7,
) -> Optional[torch.Tensor]:
    matrix = _symmetrize(matrix)
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + jitter * eye)
            return torch.cholesky_inverse(chol)
        except RuntimeError:
            jitter = base_jitter if jitter == 0.0 else jitter * 10.0
    return None


@torch.no_grad()
def extract_last_layer_features_and_probs(
    model,
    loader,
    device: Optional[torch.device] = None,
    progress_logger=None,
    tag: str = "BAIT",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    feats = []
    probs = []
    indices = []
    total_batches = len(loader)
    t0 = time.perf_counter()

    for batch_idx, (images, _, batch_indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        logits, h = model(images, return_features=True)
        p = F.softmax(logits, dim=1)

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


def _augment_features(features: torch.Tensor, use_bias: bool) -> torch.Tensor:
    if not use_bias:
        return features
    ones = torch.ones((features.size(0), 1), dtype=features.dtype, device=features.device)
    return torch.cat([features, ones], dim=1)


def build_class_cov_factors(probs: torch.Tensor, eig_floor: float = 1e-12) -> torch.Tensor:
    """
    Build L_x with shape [N, C, C] such that:
      C(p_x) = diag(p_x) - p_x p_x^T = L_x L_x^T.
    """
    cov = torch.diag_embed(probs) - probs.unsqueeze(2) * probs.unsqueeze(1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    clamped = torch.clamp(eigvals, min=0.0)
    sqrt_vals = torch.sqrt(clamped)
    sqrt_vals = torch.where(eigvals > eig_floor, sqrt_vals, torch.zeros_like(sqrt_vals))
    return eigvecs * sqrt_vals.unsqueeze(1)


def build_classification_fisher_factor(feature: torch.Tensor, class_cov_factor: torch.Tensor) -> torch.Tensor:
    """
    Construct V_x for classification last layer.
    Parameter vectorization is class-major:
      vec(W) = [w_1; ...; w_C], w_c in R^d.
    Then:
      I(x) = C(p_x) kron (phi_x phi_x^T) = V_x V_x^T.
    """
    return torch.einsum("cr,d->cdr", class_cov_factor, feature).reshape(
        class_cov_factor.size(0) * feature.numel(),
        class_cov_factor.size(1),
    )


def accumulate_fisher_from_factors(
    features: torch.Tensor,
    class_cov_factors: torch.Tensor,
    progress_logger=None,
    tag: str = "fisher",
) -> torch.Tensor:
    num_samples = features.size(0)
    if num_samples == 0:
        dim = features.size(1) * class_cov_factors.size(1)
        return torch.zeros((dim, dim), dtype=features.dtype, device=features.device)

    dim = features.size(1) * class_cov_factors.size(1)
    fisher = torch.zeros((dim, dim), dtype=features.dtype, device=features.device)
    t0 = time.perf_counter()
    for i in range(num_samples):
        factor = build_classification_fisher_factor(features[i], class_cov_factors[i])
        fisher += factor @ factor.t()
        if progress_logger is not None and ((i + 1) % 500 == 0 or (i + 1) == num_samples):
            elapsed = time.perf_counter() - t0
            avg = elapsed / float(i + 1)
            eta = avg * float(num_samples - (i + 1))
            progress_logger.log(
                f"[BAIT] {tag} {i + 1}/{num_samples} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                device=str(features.device),
            )
    return fisher / float(num_samples)


def woodbury_trace_gain(
    M_inv: torch.Tensor,
    I_u: torch.Tensor,
    factor: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Forward score gain:
      tr(V^T M^{-1} I_U M^{-1} V (I + V^T M^{-1} V)^{-1}).
    """
    U = M_inv @ factor
    A = _symmetrize(torch.eye(factor.size(1), dtype=M_inv.dtype, device=M_inv.device) + factor.t() @ U)
    A_inv = _inverse_spd_with_jitter(A)
    core = U.t() @ (I_u @ U)
    gain = float(torch.trace(core @ A_inv).item())
    return gain, U, A_inv


def update_inverse_with_factor(M_inv: torch.Tensor, U: torch.Tensor, A_inv: torch.Tensor) -> torch.Tensor:
    return _symmetrize(M_inv - U @ A_inv @ U.t())


def woodbury_trace_increase_downdate(
    M_inv: torch.Tensor,
    I_u: torch.Tensor,
    factor: torch.Tensor,
    pd_floor: float = 1e-10,
) -> Optional[Tuple[float, torch.Tensor, torch.Tensor]]:
    """
    Backward score increase:
      tr(V^T M^{-1} I_U M^{-1} V (I - V^T M^{-1} V)^{-1}).
    Returns None when the downdate is not numerically valid.
    """
    U = M_inv @ factor
    A = _symmetrize(torch.eye(factor.size(1), dtype=M_inv.dtype, device=M_inv.device) - factor.t() @ U)
    eigvals = torch.linalg.eigvalsh(A)
    min_eig = float(eigvals.min().item())
    if min_eig < -1e-8:
        return None

    jitter = max(0.0, pd_floor - min_eig)
    if jitter > 0.0:
        A = A + jitter * torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    try:
        chol = torch.linalg.cholesky(A)
        A_inv = torch.cholesky_inverse(chol)
    except RuntimeError:
        return None

    core = U.t() @ (I_u @ U)
    increase = float(torch.trace(core @ A_inv).item())
    return increase, U, A_inv


def downdate_inverse_with_factor(M_inv: torch.Tensor, U: torch.Tensor, A_inv: torch.Tensor) -> torch.Tensor:
    return _symmetrize(M_inv + U @ A_inv @ U.t())


def bait_forward_greedy(
    features: torch.Tensor,
    class_cov_factors: torch.Tensor,
    M_inv: torch.Tensor,
    I_u: torch.Tensor,
    oversample_count: int,
    progress_logger=None,
) -> Tuple[list, torch.Tensor, Dict[str, list]]:
    num_unlabeled = features.size(0)
    available = np.ones(num_unlabeled, dtype=bool)
    selected = []
    gains = []

    for step in range(oversample_count):
        candidate_indices = np.where(available)[0]
        if candidate_indices.size == 0:
            break

        best_gain = -float("inf")
        best_idx = None
        best_terms = None

        for idx in candidate_indices:
            factor = build_classification_fisher_factor(features[int(idx)], class_cov_factors[int(idx)])
            gain, U, A_inv = woodbury_trace_gain(M_inv=M_inv, I_u=I_u, factor=factor)
            if gain > best_gain:
                best_gain = gain
                best_idx = int(idx)
                best_terms = (U, A_inv)

        if best_idx is None or best_terms is None:
            break

        M_inv = update_inverse_with_factor(M_inv, best_terms[0], best_terms[1])
        available[best_idx] = False
        selected.append(best_idx)
        gains.append(best_gain)

        if progress_logger is not None and (((step + 1) % 10 == 0) or (step + 1 == oversample_count)):
            progress_logger.log(
                f"[BAIT] forward {step + 1}/{oversample_count} gain={best_gain:.6f}",
                device=str(features.device),
            )

    return selected, M_inv, {"forward_gains": gains}


def bait_backward_prune(
    features: torch.Tensor,
    class_cov_factors: torch.Tensor,
    forward_selected: list,
    target_budget: int,
    M_inv: torch.Tensor,
    I_u: torch.Tensor,
    progress_logger=None,
) -> Tuple[list, torch.Tensor, Dict[str, list]]:
    current = list(forward_selected)
    increases = []
    removed = []

    while len(current) > target_budget:
        best_idx = None
        best_increase = float("inf")
        best_terms = None

        for idx in current:
            factor = build_classification_fisher_factor(features[int(idx)], class_cov_factors[int(idx)])
            candidate = woodbury_trace_increase_downdate(M_inv=M_inv, I_u=I_u, factor=factor)
            if candidate is None:
                continue
            increase, U, A_inv = candidate
            if increase < best_increase:
                best_increase = increase
                best_idx = int(idx)
                best_terms = (U, A_inv)

        if best_idx is None or best_terms is None:
            # Robust fallback: direct check among removable points.
            M_current = _inverse_spd_with_jitter(M_inv)
            fallback_best = None
            fallback_obj = float("inf")
            fallback_inv = None
            for idx in current:
                factor = build_classification_fisher_factor(features[int(idx)], class_cov_factors[int(idx)])
                M_candidate = _symmetrize(M_current - factor @ factor.t())
                cand_inv = _try_inverse_spd_with_jitter(M_candidate)
                if cand_inv is None:
                    continue
                obj = float(torch.trace(cand_inv @ I_u).item())
                if obj < fallback_obj:
                    fallback_obj = obj
                    fallback_best = int(idx)
                    fallback_inv = cand_inv
            if fallback_best is None or fallback_inv is None:
                raise RuntimeError(
                    "BAIT backward pruning failed: no numerically valid downdate candidate. "
                    "Try increasing --bait_lambda."
                )
            best_idx = fallback_best
            best_increase = fallback_obj
            M_inv = _symmetrize(fallback_inv)
        else:
            M_inv = downdate_inverse_with_factor(M_inv, best_terms[0], best_terms[1])

        current.remove(best_idx)
        removed.append(best_idx)
        increases.append(best_increase)

        if progress_logger is not None and ((len(current) % 10 == 0) or (len(current) == target_budget)):
            progress_logger.log(
                f"[BAIT] backward keep={len(current)} increase={best_increase:.6f}",
                device=str(features.device),
            )

    return current, M_inv, {"backward_increases": increases, "removed_indices": removed}


def select_bait(
    features: torch.Tensor,
    probs: torch.Tensor,
    labeled_features: torch.Tensor,
    labeled_probs: torch.Tensor,
    budget: int,
    lambda_reg: float,
    oversample_factor: int,
    use_bias: bool,
    device: torch.device,
    progress_logger=None,
):
    if features.size(0) == 0 or budget <= 0:
        return np.array([], dtype=np.int64), {"forward_gains": [], "backward_increases": []}

    fisher_dtype = torch.float64

    features = features.to(device=device, dtype=fisher_dtype)
    probs = probs.to(device=device, dtype=fisher_dtype)
    labeled_features = labeled_features.to(device=device, dtype=fisher_dtype)
    labeled_probs = labeled_probs.to(device=device, dtype=fisher_dtype)

    features = _augment_features(features, use_bias=use_bias)
    labeled_features = _augment_features(labeled_features, use_bias=use_bias)

    unlabeled_cov_factors = build_class_cov_factors(probs)
    labeled_cov_factors = build_class_cov_factors(labeled_probs)

    t0 = time.perf_counter()
    I_u = accumulate_fisher_from_factors(
        features=features,
        class_cov_factors=unlabeled_cov_factors,
        progress_logger=progress_logger,
        tag="I_U",
    )
    pool_fisher_time = time.perf_counter() - t0
    I_labeled = accumulate_fisher_from_factors(
        features=labeled_features,
        class_cov_factors=labeled_cov_factors,
        progress_logger=progress_logger,
        tag="I_labeled",
    )

    dim = I_u.size(0)
    M0 = I_labeled + float(lambda_reg) * torch.eye(dim, dtype=fisher_dtype, device=device)
    M_inv = _inverse_spd_with_jitter(M0)

    effective_budget = min(int(budget), int(features.size(0)))
    oversample_target = min(max(effective_budget, int(oversample_factor) * effective_budget), int(features.size(0)))

    selected_forward, M_inv_after_forward, forward_debug = bait_forward_greedy(
        features=features,
        class_cov_factors=unlabeled_cov_factors,
        M_inv=M_inv,
        I_u=I_u,
        oversample_count=oversample_target,
        progress_logger=progress_logger,
    )

    selected_final, _, backward_debug = bait_backward_prune(
        features=features,
        class_cov_factors=unlabeled_cov_factors,
        forward_selected=selected_forward,
        target_budget=effective_budget,
        M_inv=M_inv_after_forward,
        I_u=I_u,
        progress_logger=progress_logger,
    )

    debug = {
        **forward_debug,
        **backward_debug,
        "pool_fisher_time_sec": float(pool_fisher_time),
        "fisher_dim": int(dim),
        "num_classes": int(probs.size(1)),
        "feature_dim_with_bias": int(features.size(1)),
        "oversample_target": int(oversample_target),
        "lambda_reg": float(lambda_reg),
        "use_bias": bool(use_bias),
    }
    return np.asarray(selected_final, dtype=np.int64), debug


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
        unlabeled_features, unlabeled_probs, _ = extract_last_layer_features_and_probs(
            model=model,
            loader=unlabeled_loader,
            device=device,
            progress_logger=progress_logger,
            tag="BAIT-unlabeled",
        )
        labeled_features, labeled_probs, _ = extract_last_layer_features_and_probs(
            model=model,
            loader=labeled_loader,
            device=device,
            progress_logger=progress_logger,
            tag="BAIT-labeled",
        )
        scoring_time = time.perf_counter() - t_score

        t_sel = time.perf_counter()
        picked_local, debug = select_bait(
            features=unlabeled_features,
            probs=unlabeled_probs,
            labeled_features=labeled_features,
            labeled_probs=labeled_probs,
            budget=budget,
            lambda_reg=self.cfg.bait_lambda,
            oversample_factor=self.cfg.bait_oversample_factor,
            use_bias=self.cfg.bait_use_bias,
            device=device,
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

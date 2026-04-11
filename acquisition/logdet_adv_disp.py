import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition, clamp_to_valid_range, scaled_linf_eps, tensor_stats


def _iter_unlabeled_batches(
    unlabeled_loader,
    images: Optional[torch.Tensor],
    tensor_batch_size: Optional[int],
):
    if unlabeled_loader is None and images is None:
        raise ValueError("Either unlabeled_loader or images must be provided.")
    if unlabeled_loader is not None and images is not None:
        raise ValueError("Provide only one of unlabeled_loader or images.")

    if unlabeled_loader is not None:
        total_batches = len(unlabeled_loader)
        for batch in unlabeled_loader:
            yield batch[0], total_batches
        return

    n = int(images.size(0))
    bs = n if tensor_batch_size is None else max(1, int(tensor_batch_size))
    total_batches = int(math.ceil(float(n) / float(bs)))
    for start in range(0, n, bs):
        end = min(start + bs, n)
        yield images[start:end], total_batches


def _fgsm_logit_displacement_attack(
    model,
    x0: torch.Tensor,
    clean_logits: torch.Tensor,
    eps_t: torch.Tensor,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> torch.Tensor:
    x_adv = x0.detach().clone().requires_grad_(True)
    adv_logits = model(x_adv)
    displacement_obj = (adv_logits - clean_logits).pow(2).sum(dim=1).mean()
    grad = torch.autograd.grad(displacement_obj, x_adv, only_inputs=True)[0]
    delta = torch.clamp(eps_t * grad.sign(), min=-eps_t, max=eps_t)
    return clamp_to_valid_range(x0 + delta, mean=mean, std=std).detach()


def _extract_logits(forward_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> torch.Tensor:
    if isinstance(forward_output, (tuple, list)):
        if len(forward_output) == 0:
            raise ValueError("Model forward returned empty tuple/list.")
        return forward_output[0]
    return forward_output


def forward_with_features(
    model,
    x: torch.Tensor,
    require_features: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Unified helper to fetch penultimate features and logits.
    Supports models exposing either:
    - forward(x, return_features=True) -> (logits, features)
    - forward_features + forward_classifier
    """
    features = None
    logits = None

    try:
        out = model(x, return_features=True)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits = out[0]
            features = out[1]
            return features, logits
        logits = _extract_logits(out)
    except TypeError:
        pass

    if features is None and hasattr(model, "forward_features") and hasattr(model, "forward_classifier"):
        features = model.forward_features(x)
        logits = model.forward_classifier(features)
        return features, logits

    if logits is None:
        logits = _extract_logits(model(x))

    if require_features and features is None:
        raise RuntimeError(
            "Could not extract penultimate features from model. "
            "Expected forward(return_features=True) or forward_features/forward_classifier."
        )
    return features, logits


def _fgsm_predictive_ce_attack(
    model,
    x0: torch.Tensor,
    pseudo_labels: torch.Tensor,
    eps_t: torch.Tensor,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> torch.Tensor:
    x_adv = x0.detach().clone().requires_grad_(True)
    logits = _extract_logits(model(x_adv))
    ce_obj = F.cross_entropy(logits, pseudo_labels, reduction="mean")
    grad = torch.autograd.grad(ce_obj, x_adv, only_inputs=True)[0]
    delta = torch.clamp(eps_t * grad.sign(), min=-eps_t, max=eps_t)
    return clamp_to_valid_range(x0 + delta, mean=mean, std=std).detach()


def _pgd_predictive_ce_attack(
    model,
    x0: torch.Tensor,
    pseudo_labels: torch.Tensor,
    eps_t: torch.Tensor,
    alpha_t: torch.Tensor,
    steps: int,
    random_start: bool,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> torch.Tensor:
    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
        x_adv = clamp_to_valid_range(x0 + delta, mean=mean, std=std)
    else:
        x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = _extract_logits(model(x_adv))
        ce_obj = F.cross_entropy(logits, pseudo_labels, reduction="mean")
        grad = torch.autograd.grad(ce_obj, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = clamp_to_valid_range(x0 + delta, mean=mean, std=std)
    return x_adv.detach()


def _pgd_logit_displacement_attack(
    model,
    x0: torch.Tensor,
    clean_logits: torch.Tensor,
    eps_t: torch.Tensor,
    alpha_t: torch.Tensor,
    steps: int,
    random_start: bool,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> torch.Tensor:
    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
        x_adv = clamp_to_valid_range(x0 + delta, mean=mean, std=std)
    else:
        x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        adv_logits = model(x_adv)
        displacement_obj = (adv_logits - clean_logits).pow(2).sum(dim=1).mean()
        grad = torch.autograd.grad(displacement_obj, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = clamp_to_valid_range(x0 + delta, mean=mean, std=std)

    return x_adv.detach()


def compute_adv_displacement_embeddings(
    model,
    unlabeled_loader=None,
    images: Optional[torch.Tensor] = None,
    attack_type: str = "fgsm",
    attack_norm: str = "linf",
    epsilon: float = 1.0 / 255.0,
    pgd_steps: int = 5,
    pgd_step_size: Optional[float] = None,
    pgd_random_start: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    progress_logger=None,
    progress_method_name: str = "LOGDET_ADV_DISP",
    tensor_batch_size: Optional[int] = None,
    return_clean_logits: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute adversarial semantic displacement vectors:
      Delta(x) = z(x_adv) - z(x)
    where x_adv approximately maximizes:
      ||z(x_adv) - z(x)||_2^2 under ||x_adv - x||_inf <= epsilon.

    This supports both dataloader input and direct tensor input.
    """
    if device is None:
        device = next(model.parameters()).device

    if float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if attack_norm.lower() != "linf":
        raise ValueError(f"Only linf attack_norm is supported, got {attack_norm}")

    attack_type = attack_type.lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    if attack_type == "pgd" and int(pgd_steps) <= 0:
        raise ValueError(f"pgd_steps must be positive, got {pgd_steps}")

    was_training = model.training
    model.eval()

    displacements = []
    clean_logits_all = [] if bool(return_clean_logits) else None
    t0 = time.perf_counter()

    for batch_idx, (batch_images, total_batches) in enumerate(
        _iter_unlabeled_batches(
            unlabeled_loader=unlabeled_loader,
            images=images,
            tensor_batch_size=tensor_batch_size,
        ),
        start=1,
    ):
        x0 = batch_images.to(device, non_blocking=True).detach()
        channels = int(x0.size(1))
        eps_t = scaled_linf_eps(
            epsilon=float(epsilon),
            std=std,
            device=x0.device,
            dtype=x0.dtype,
            channels=channels,
        )

        with torch.no_grad():
            clean_logits = model(x0).detach()
            if clean_logits_all is not None:
                clean_logits_all.append(clean_logits.to(dtype=torch.float32))

        if attack_type == "fgsm":
            x_adv = _fgsm_logit_displacement_attack(
                model=model,
                x0=x0,
                clean_logits=clean_logits,
                eps_t=eps_t,
                mean=mean,
                std=std,
            )
        else:
            steps = int(pgd_steps)
            if pgd_step_size is None:
                step_size = float(epsilon) / max(float(steps) / 2.0, 1.0)
            else:
                step_size = float(pgd_step_size)
            if step_size <= 0.0:
                raise ValueError(f"pgd_step_size must be positive, got {step_size}")
            alpha_t = scaled_linf_eps(
                epsilon=step_size,
                std=std,
                device=x0.device,
                dtype=x0.dtype,
                channels=channels,
            )
            x_adv = _pgd_logit_displacement_attack(
                model=model,
                x0=x0,
                clean_logits=clean_logits,
                eps_t=eps_t,
                alpha_t=alpha_t,
                steps=steps,
                random_start=bool(pgd_random_start),
                mean=mean,
                std=std,
            )

        with torch.no_grad():
            adv_logits = model(x_adv)
            displacements.append((adv_logits - clean_logits).to(dtype=torch.float32))

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=progress_method_name,
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    if was_training:
        model.train()

    if len(displacements) == 0:
        empty = torch.empty((0, 0), dtype=torch.float32, device=device)
        if clean_logits_all is not None:
            return empty, empty
        return empty
    disp_cat = torch.cat(displacements, dim=0)
    if clean_logits_all is not None:
        return disp_cat, torch.cat(clean_logits_all, dim=0)
    return disp_cat


def compute_adv_q_scores_from_logit_displacements(
    displacements: torch.Tensor,
    clean_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute adversarial viability scores from cached logit displacement vectors.

    q_t(x) = CE(z(x + delta*(x)), y_hat(x)), where y_hat(x) is the clean
    prediction and delta*(x) is the same logit-displacement attack used to form
    Delta(x) = z(x + delta*(x)) - z(x).
    """
    if displacements.shape != clean_logits.shape:
        raise ValueError(
            "displacements and clean_logits must have the same shape, "
            f"got {tuple(displacements.shape)} and {tuple(clean_logits.shape)}"
        )
    if displacements.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=clean_logits.device)
    pseudo_labels = clean_logits.argmax(dim=1)
    adv_logits = clean_logits + displacements
    return F.cross_entropy(adv_logits, pseudo_labels, reduction="none").to(dtype=torch.float32)


def top_retained_local_by_q(
    q_scores: torch.Tensor,
    retain_fraction: float,
) -> torch.Tensor:
    """Return local row indices for the top ceil(retain_fraction * N) q scores."""
    if not (0.0 < float(retain_fraction) <= 1.0):
        raise ValueError(f"retain_fraction must be in (0, 1], got {retain_fraction}")
    n = int(q_scores.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=q_scores.device)
    retain_k = int(math.ceil(float(retain_fraction) * float(n)))
    retain_k = max(1, min(retain_k, n))
    scores = torch.where(torch.isfinite(q_scores), q_scores, torch.full_like(q_scores, -torch.inf))
    return torch.topk(scores, k=retain_k, largest=True).indices


def compute_adv_semantic_displacement_embeddings(
    model,
    unlabeled_loader=None,
    images: Optional[torch.Tensor] = None,
    embedding_space: str = "features",
    attack_type: str = "fgsm",
    attack_norm: str = "linf",
    epsilon: float = 1.0 / 255.0,
    pgd_steps: int = 5,
    pgd_step_size: Optional[float] = None,
    pgd_random_start: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    progress_logger=None,
    progress_method_name: str = "LOGDET_ADV_SEMANTIC",
    tensor_batch_size: Optional[int] = None,
    return_clean_logits: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Label-aware adversarial semantic displacements with predicted-label CE attack.

    y_hat(x) = argmax z(x)
    delta* approximately maximizes CE(z(x+delta), y_hat) under ||delta||_inf <= epsilon

    Displacement:
      - features: f(x+delta*) - f(x)
      - logits:   z(x+delta*) - z(x)
    """
    if device is None:
        device = next(model.parameters()).device

    if float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if attack_norm.lower() != "linf":
        raise ValueError(f"Only linf attack_norm is supported, got {attack_norm}")

    embedding_space = embedding_space.lower()
    if embedding_space not in {"features", "logits"}:
        raise ValueError(f"embedding_space must be one of ['features','logits'], got {embedding_space}")

    attack_type = attack_type.lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type: {attack_type}")
    if attack_type == "pgd" and int(pgd_steps) <= 0:
        raise ValueError(f"pgd_steps must be positive, got {pgd_steps}")

    was_training = model.training
    model.eval()

    displacements = []
    clean_logits_all = [] if bool(return_clean_logits) else None
    t0 = time.perf_counter()

    for batch_idx, (batch_images, total_batches) in enumerate(
        _iter_unlabeled_batches(
            unlabeled_loader=unlabeled_loader,
            images=images,
            tensor_batch_size=tensor_batch_size,
        ),
        start=1,
    ):
        x0 = batch_images.to(device, non_blocking=True).detach()
        channels = int(x0.size(1))
        eps_t = scaled_linf_eps(
            epsilon=float(epsilon),
            std=std,
            device=x0.device,
            dtype=x0.dtype,
            channels=channels,
        )

        with torch.no_grad():
            clean_features, clean_logits = forward_with_features(
                model=model,
                x=x0,
                require_features=(embedding_space == "features"),
            )
            pseudo_labels = torch.argmax(clean_logits, dim=1)
            if clean_logits_all is not None:
                clean_logits_all.append(clean_logits.to(dtype=torch.float32))

        if attack_type == "fgsm":
            x_adv = _fgsm_predictive_ce_attack(
                model=model,
                x0=x0,
                pseudo_labels=pseudo_labels,
                eps_t=eps_t,
                mean=mean,
                std=std,
            )
        else:
            steps = int(pgd_steps)
            if pgd_step_size is None:
                step_size = float(epsilon) / max(float(steps) / 2.0, 1.0)
            else:
                step_size = float(pgd_step_size)
            if step_size <= 0.0:
                raise ValueError(f"pgd_step_size must be positive, got {step_size}")
            alpha_t = scaled_linf_eps(
                epsilon=step_size,
                std=std,
                device=x0.device,
                dtype=x0.dtype,
                channels=channels,
            )
            x_adv = _pgd_predictive_ce_attack(
                model=model,
                x0=x0,
                pseudo_labels=pseudo_labels,
                eps_t=eps_t,
                alpha_t=alpha_t,
                steps=steps,
                random_start=bool(pgd_random_start),
                mean=mean,
                std=std,
            )

        with torch.no_grad():
            adv_features, adv_logits = forward_with_features(
                model=model,
                x=x_adv,
                require_features=(embedding_space == "features"),
            )
            if embedding_space == "features":
                delta = adv_features - clean_features
            else:
                delta = adv_logits - clean_logits
            displacements.append(delta.to(dtype=torch.float32))

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=progress_method_name,
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    if was_training:
        model.train()

    if len(displacements) == 0:
        empty = torch.empty((0, 0), dtype=torch.float32, device=device)
        if clean_logits_all is not None:
            return empty, empty
        return empty

    disp_cat = torch.cat(displacements, dim=0)
    if clean_logits_all is not None:
        return disp_cat, torch.cat(clean_logits_all, dim=0)
    return disp_cat


def _inverse_spd_with_jitter(
    matrix: torch.Tensor,
    base_jitter: float = 1e-8,
    max_tries: int = 6,
) -> Tuple[torch.Tensor, float, bool]:
    matrix = 0.5 * (matrix + matrix.t())
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + jitter * eye)
            return torch.cholesky_inverse(chol), float(jitter), False
        except RuntimeError:
            jitter = float(base_jitter) if jitter == 0.0 else float(jitter * 10.0)
    pinv = torch.linalg.pinv(matrix + max(float(base_jitter), float(jitter)) * eye)
    return pinv, float(max(float(base_jitter), float(jitter))), True


def _quadratic_form_scores(
    displacements: torch.Tensor,
    a_inv: torch.Tensor,
    score_chunk_size: int,
) -> torch.Tensor:
    n = int(displacements.size(0))
    if score_chunk_size <= 0 or n <= score_chunk_size:
        proj = displacements @ a_inv
        return (proj * displacements).sum(dim=1)

    out = torch.empty((n,), dtype=displacements.dtype, device=displacements.device)
    for start in range(0, n, score_chunk_size):
        end = min(start + score_chunk_size, n)
        chunk = displacements[start:end]
        proj = chunk @ a_inv
        out[start:end] = (proj * chunk).sum(dim=1)
    return out


def _logdet_spd(
    matrix: torch.Tensor,
    base_jitter: float = 1e-8,
    max_tries: int = 6,
) -> Tuple[float, float, bool]:
    matrix = 0.5 * (matrix + matrix.t())
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + jitter * eye)
            logdet = float((2.0 * torch.log(torch.diag(chol))).sum().item())
            return logdet, float(jitter), False
        except RuntimeError:
            jitter = float(base_jitter) if jitter == 0.0 else float(jitter * 10.0)

    fallback_jitter = max(float(base_jitter), float(jitter))
    sign, logabsdet = torch.linalg.slogdet(matrix + fallback_jitter * eye)
    if float(sign.item()) <= 0.0:
        return float("-inf"), float(fallback_jitter), True
    return float(logabsdet.item()), float(fallback_jitter), True


def _rebuild_selected_state(
    displacements: torch.Tensor,
    selected_local: torch.Tensor,
    lambda_reg: float,
    jitter: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, Any]]:
    c = int(displacements.size(1))
    eye = torch.eye(c, dtype=displacements.dtype, device=displacements.device)
    if selected_local.numel() == 0:
        selected_tensor = torch.empty((0, c), dtype=displacements.dtype, device=displacements.device)
    else:
        selected_tensor = displacements[selected_local]

    a = float(lambda_reg) * eye + selected_tensor.t() @ selected_tensor
    a_inv, inv_jitter, inv_used_pinv = _inverse_spd_with_jitter(
        matrix=a,
        base_jitter=float(jitter),
        max_tries=6,
    )
    obj, logdet_jitter, logdet_fallback = _logdet_spd(
        matrix=a,
        base_jitter=float(jitter),
        max_tries=6,
    )
    return a, a_inv, obj, {
        "inv_jitter": float(inv_jitter),
        "inv_used_pinv": bool(inv_used_pinv),
        "logdet_jitter": float(logdet_jitter),
        "logdet_fallback": bool(logdet_fallback),
    }


def greedy_logdet_selector(
    displacements: torch.Tensor,
    query_size: int,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    progress_logger=None,
    progress_method_name: str = "LOGDET_ADV_DISP_GREEDY",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Greedy maximization of:
      log det(lambda I + sum_{x in B} Delta(x) Delta(x)^T)
    using Sherman-Morrison rank-1 inverse updates.
    """
    if displacements.ndim != 2:
        raise ValueError(f"displacements must be rank-2 [N,C], got shape={tuple(displacements.shape)}")
    if float(lambda_reg) <= 0.0:
        raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
    if float(jitter) <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    d = displacements.to(dtype=torch.float32)
    n, c = int(d.size(0)), int(d.size(1))
    k = min(int(query_size), n)

    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64), {
            "selected_scores": [],
            "selected_log_marginal_gains": [],
            "nonfinite_score_steps": 0,
            "inverse_rebuilds": 0,
            "used_pinv_rebuilds": 0,
            "max_jitter_used": 0.0,
        }

    eye = torch.eye(c, dtype=d.dtype, device=d.device)
    a_inv = eye / float(lambda_reg)

    selected_mask = torch.zeros((n,), dtype=torch.bool, device=d.device)
    selected = []
    selected_scores = []
    selected_log_marginal_gains = []

    nonfinite_score_steps = 0
    inverse_rebuilds = 0
    used_pinv_rebuilds = 0
    max_jitter_used = 0.0

    for step in range(k):
        scores = _quadratic_form_scores(d, a_inv, int(score_chunk_size))
        finite_scores = torch.isfinite(scores)
        if not bool(torch.all(finite_scores)):
            nonfinite_score_steps += 1
            scores = torch.where(finite_scores, scores, torch.full_like(scores, -torch.inf))
        scores = scores.masked_fill(selected_mask, -torch.inf)

        best_local = int(torch.argmax(scores).item())
        best_score = float(scores[best_local].item())
        if not math.isfinite(best_score):
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
            if remaining.numel() == 0:
                break
            best_local = int(remaining[0].item())
            u_fallback = d[best_local]
            best_score = float(torch.dot(u_fallback, a_inv @ u_fallback).item())

        u = d[best_local]
        v = a_inv @ u
        denom = float(1.0 + torch.dot(u, v).item())

        if not math.isfinite(denom) or denom <= float(jitter):
            inverse_rebuilds += 1
            selected_mask[best_local] = True
            selected.append(best_local)
            selected_scores.append(best_score)
            selected_log_marginal_gains.append(float(np.log1p(max(best_score, 0.0))))

            selected_tensor = d[selected]
            a = float(lambda_reg) * eye + selected_tensor.t() @ selected_tensor
            a_inv, used_jitter, used_pinv = _inverse_spd_with_jitter(
                matrix=a,
                base_jitter=float(jitter),
                max_tries=6,
            )
            max_jitter_used = max(max_jitter_used, float(used_jitter))
            if used_pinv:
                used_pinv_rebuilds += 1
        else:
            a_inv = a_inv - torch.outer(v, v) / denom
            a_inv = 0.5 * (a_inv + a_inv.t())
            selected_mask[best_local] = True
            selected.append(best_local)
            selected_scores.append(best_score)
            selected_log_marginal_gains.append(float(np.log1p(max(best_score, 0.0))))

        if progress_logger is not None and ((step + 1) % 10 == 0 or (step + 1) == k):
            progress_logger.log(
                (
                    f"[{progress_method_name}] step={step + 1}/{k} "
                    f"best_score={best_score:.6f} "
                    f"rebuilds={inverse_rebuilds}"
                ),
                device=str(d.device),
            )

    return np.asarray(selected, dtype=np.int64), {
        "selected_scores": [float(x) for x in selected_scores],
        "selected_log_marginal_gains": [float(x) for x in selected_log_marginal_gains],
        "initial_logdet_objective": float(c * math.log(float(lambda_reg))),
        "final_logdet_objective": float(c * math.log(float(lambda_reg)) + float(np.sum(selected_log_marginal_gains))),
        "nonfinite_score_steps": int(nonfinite_score_steps),
        "inverse_rebuilds": int(inverse_rebuilds),
        "used_pinv_rebuilds": int(used_pinv_rebuilds),
        "max_jitter_used": float(max_jitter_used),
    }


def refine_logdet_swaps(
    displacements: torch.Tensor,
    selected_local: np.ndarray,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    max_swap_rounds: int = 3,
    swap_top_unselected: int = 200,
    swap_top_selected: int = 0,
    swap_improvement_tol: float = 1e-8,
    swap_downdate_tol: float = 1e-6,
    progress_logger=None,
    progress_method_name: str = "LOGDET_ADV_DISP_SWAP",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    1-swap local refinement for:
      F(S) = log det(lambda I + sum_{i in S} Delta_i Delta_i^T)

    Starting from a greedy set S, repeatedly accept the best improving swap
    (remove b in S, add u not in S) until convergence or max rounds.
    """
    if displacements.ndim != 2:
        raise ValueError(f"displacements must be rank-2 [N,C], got shape={tuple(displacements.shape)}")
    if float(lambda_reg) <= 0.0:
        raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
    if float(jitter) <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    d = displacements.to(dtype=torch.float32)
    n = int(d.size(0))
    selected = torch.as_tensor(selected_local, dtype=torch.long, device=d.device)
    if selected.numel() == 0 or n == 0:
        return np.asarray([], dtype=np.int64), {
            "initial_logdet_objective": float("nan"),
            "final_logdet_objective": float("nan"),
            "accepted_swaps": 0,
            "swap_rounds_run": 0,
            "best_swap_gains_by_round": [],
            "accepted_swap_gains": [],
            "downdate_fallback_rebuilds": 0,
            "downdate_skips": 0,
            "state_rebuilds": 0,
            "state_used_pinv_rebuilds": 0,
            "max_jitter_used": 0.0,
            "nonfinite_swap_score_events": 0,
        }

    selected = torch.unique(selected, sorted=False)
    selected_mask = torch.zeros((n,), dtype=torch.bool, device=d.device)
    selected_mask[selected] = True

    _, a_inv, current_obj, init_state_debug = _rebuild_selected_state(
        displacements=d,
        selected_local=selected,
        lambda_reg=float(lambda_reg),
        jitter=float(jitter),
    )
    initial_obj = float(current_obj)

    max_jitter_used = max(float(init_state_debug["inv_jitter"]), float(init_state_debug["logdet_jitter"]))
    state_rebuilds = 1
    state_used_pinv_rebuilds = int(init_state_debug["inv_used_pinv"])
    nonfinite_swap_score_events = 0
    downdate_fallback_rebuilds = 0
    downdate_skips = 0
    accepted_swaps = 0
    best_swap_gains_by_round: List[float] = []
    accepted_swap_gains: List[float] = []

    rounds = max(0, int(max_swap_rounds))
    for swap_round in range(rounds):
        unselected = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
        if unselected.numel() == 0 or selected.numel() == 0:
            break

        # Candidate pruning: top-L unselected by current quadratic score.
        if int(swap_top_unselected) > 0 and int(unselected.numel()) > int(swap_top_unselected):
            unselected_scores = _quadratic_form_scores(d[unselected], a_inv, int(score_chunk_size))
            unselected_scores = torch.where(
                torch.isfinite(unselected_scores),
                unselected_scores,
                torch.full_like(unselected_scores, -torch.inf),
            )
            top_u = min(int(swap_top_unselected), int(unselected.numel()))
            top_u_idx = torch.topk(unselected_scores, k=top_u, largest=True).indices
            candidate_unselected = unselected[top_u_idx]
        else:
            candidate_unselected = unselected

        # Candidate pruning: evaluate worst-L selected (small quadratic score) or all.
        if int(swap_top_selected) > 0 and int(selected.numel()) > int(swap_top_selected):
            selected_scores = _quadratic_form_scores(d[selected], a_inv, int(score_chunk_size))
            selected_scores = torch.where(
                torch.isfinite(selected_scores),
                selected_scores,
                torch.full_like(selected_scores, torch.inf),
            )
            top_b = min(int(swap_top_selected), int(selected.numel()))
            worst_sel_idx = torch.topk(selected_scores, k=top_b, largest=False).indices
            candidate_selected = selected[worst_sel_idx]
        else:
            candidate_selected = selected

        d_unselected = d[candidate_unselected]
        best_gain = float("-inf")
        best_remove = None
        best_add = None

        selected_list = selected.tolist()
        selected_pos = {int(idx): pos for pos, idx in enumerate(selected_list)}

        for b_local in candidate_selected.tolist():
            b_local = int(b_local)
            u_b = d[b_local]
            v_b = a_inv @ u_b
            r_b = float(torch.dot(u_b, v_b).item())
            one_minus = float(1.0 - r_b)

            use_direct_fallback = (not math.isfinite(one_minus)) or (one_minus <= float(swap_downdate_tol))
            if use_direct_fallback:
                selected_wo = selected[selected != b_local]
                if selected_wo.numel() == selected.numel():
                    downdate_skips += 1
                    continue
                _, a_minus_inv, obj_minus, minus_debug = _rebuild_selected_state(
                    displacements=d,
                    selected_local=selected_wo,
                    lambda_reg=float(lambda_reg),
                    jitter=float(jitter),
                )
                downdate_fallback_rebuilds += 1
                state_used_pinv_rebuilds += int(minus_debug["inv_used_pinv"])
                max_jitter_used = max(
                    max_jitter_used,
                    float(minus_debug["inv_jitter"]),
                    float(minus_debug["logdet_jitter"]),
                )
                removal_const = float(obj_minus - current_obj)
            else:
                a_minus_inv = a_inv + torch.outer(v_b, v_b) / one_minus
                a_minus_inv = 0.5 * (a_minus_inv + a_minus_inv.t())
                removal_const = float(math.log(one_minus))

            q = _quadratic_form_scores(d_unselected, a_minus_inv, int(score_chunk_size))
            finite_q = torch.isfinite(q)
            if not bool(torch.all(finite_q)):
                nonfinite_swap_score_events += 1
            valid_q = finite_q & (q > (-1.0 + float(jitter)))
            gains = torch.full_like(q, -torch.inf)
            if bool(valid_q.any()):
                gains[valid_q] = float(removal_const) + torch.log1p(q[valid_q])

            local_best = int(torch.argmax(gains).item())
            local_best_gain = float(gains[local_best].item())
            if math.isfinite(local_best_gain) and local_best_gain > best_gain:
                best_gain = local_best_gain
                best_remove = b_local
                best_add = int(candidate_unselected[local_best].item())

        best_swap_gains_by_round.append(float(best_gain) if math.isfinite(best_gain) else float("-inf"))

        if best_remove is None or best_add is None:
            if progress_logger is not None:
                progress_logger.log(
                    f"[{progress_method_name}] round={swap_round + 1}/{rounds} no_valid_swap",
                    device=str(d.device),
                )
            break

        if best_gain <= float(swap_improvement_tol):
            if progress_logger is not None:
                progress_logger.log(
                    (
                        f"[{progress_method_name}] round={swap_round + 1}/{rounds} "
                        f"best_gain={best_gain:.6e} <= tol={float(swap_improvement_tol):.6e} (stop)"
                    ),
                    device=str(d.device),
                )
            break

        # Accept the best improving swap.
        remove_pos = selected_pos[int(best_remove)]
        selected_mask[int(best_remove)] = False
        selected_mask[int(best_add)] = True
        selected[remove_pos] = int(best_add)
        accepted_swaps += 1
        accepted_swap_gains.append(float(best_gain))

        _, a_inv, current_obj, rebuild_debug = _rebuild_selected_state(
            displacements=d,
            selected_local=selected,
            lambda_reg=float(lambda_reg),
            jitter=float(jitter),
        )
        state_rebuilds += 1
        state_used_pinv_rebuilds += int(rebuild_debug["inv_used_pinv"])
        max_jitter_used = max(
            max_jitter_used,
            float(rebuild_debug["inv_jitter"]),
            float(rebuild_debug["logdet_jitter"]),
        )

        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[{progress_method_name}] round={swap_round + 1}/{rounds} accepted "
                    f"remove={best_remove} add={best_add} gain={best_gain:.6e} "
                    f"objective={current_obj:.6f}"
                ),
                device=str(d.device),
            )

    return selected.detach().cpu().numpy().astype(np.int64), {
        "initial_logdet_objective": float(initial_obj),
        "final_logdet_objective": float(current_obj),
        "accepted_swaps": int(accepted_swaps),
        "swap_rounds_run": int(len(best_swap_gains_by_round)),
        "best_swap_gains_by_round": [float(x) for x in best_swap_gains_by_round],
        "accepted_swap_gains": [float(x) for x in accepted_swap_gains],
        "downdate_fallback_rebuilds": int(downdate_fallback_rebuilds),
        "downdate_skips": int(downdate_skips),
        "state_rebuilds": int(state_rebuilds),
        "state_used_pinv_rebuilds": int(state_used_pinv_rebuilds),
        "max_jitter_used": float(max_jitter_used),
        "nonfinite_swap_score_events": int(nonfinite_swap_score_events),
        "swap_top_unselected": int(swap_top_unselected),
        "swap_top_selected": int(swap_top_selected),
        "swap_improvement_tol": float(swap_improvement_tol),
        "swap_downdate_tol": float(swap_downdate_tol),
    }


def _assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    if x.numel() > 0 and not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains NaN/Inf values")


def _safe_step_size(epsilon: float, steps: int, step_size: Optional[float]) -> float:
    if step_size is None:
        return float(epsilon) / max(float(steps) / 2.0, 1.0)
    return float(step_size)


def compute_adv_gradient_displacement_components(
    model,
    unlabeled_loader=None,
    images: Optional[torch.Tensor] = None,
    epsilon: float = 1.0 / 255.0,
    attack_steps: int = 3,
    attack_step_size: Optional[float] = None,
    attack_random_start: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    progress_logger=None,
    progress_method_name: str = "ADV_GRAD_DISP_ATTACK",
    tensor_batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the pieces needed for adversarial BADGE-style displacement kernels.

    For clean logits z(x), penultimate features h(x), and pseudo-label
    y_hat=argmax softmax(z(x)), the clean last-layer gradient embedding is

        g_clean(x) = vec((p_clean(x) - e_y_hat) h_clean(x)^T).

    The acquisition perturbation delta* is produced by projected gradient
    ascent on ||z(x+delta)-z(x)||_2^2 under an L_inf acquisition budget.  The
    adversarial embedding reuses the clean pseudo-label:

        g_adv(x) = vec((p_adv(x) - e_y_hat) h_adv(x)^T).

    This function returns the low-dimensional factors a_clean, h_clean,
    a_adv, h_adv rather than materializing vec(.) blocks.
    """
    if device is None:
        device = next(model.parameters()).device
    if float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if int(attack_steps) <= 0:
        raise ValueError(f"attack_steps must be positive, got {attack_steps}")

    step_size = _safe_step_size(float(epsilon), int(attack_steps), attack_step_size)
    if step_size <= 0.0:
        raise ValueError(f"attack_step_size must be positive or None, got {attack_step_size}")

    was_training = model.training
    model.eval()

    clean_logits_all: List[torch.Tensor] = []
    adv_logits_all: List[torch.Tensor] = []
    clean_features_all: List[torch.Tensor] = []
    adv_features_all: List[torch.Tensor] = []
    a_clean_all: List[torch.Tensor] = []
    a_adv_all: List[torch.Tensor] = []
    y_hat_all: List[torch.Tensor] = []

    t0 = time.perf_counter()
    for batch_idx, (batch_images, total_batches) in enumerate(
        _iter_unlabeled_batches(
            unlabeled_loader=unlabeled_loader,
            images=images,
            tensor_batch_size=tensor_batch_size,
        ),
        start=1,
    ):
        x0 = batch_images.to(device, non_blocking=True).detach()
        channels = int(x0.size(1))
        eps_t = scaled_linf_eps(
            epsilon=float(epsilon),
            std=std,
            device=x0.device,
            dtype=x0.dtype,
            channels=channels,
        )
        alpha_t = scaled_linf_eps(
            epsilon=float(step_size),
            std=std,
            device=x0.device,
            dtype=x0.dtype,
            channels=channels,
        )

        with torch.no_grad():
            clean_features, clean_logits = forward_with_features(model=model, x=x0, require_features=True)
            clean_probs = F.softmax(clean_logits, dim=1)
            y_hat = clean_probs.argmax(dim=1)
            one_hot = F.one_hot(y_hat, num_classes=clean_probs.size(1)).to(dtype=clean_probs.dtype)
            a_clean = clean_probs - one_hot

        x_adv = _pgd_logit_displacement_attack(
            model=model,
            x0=x0,
            clean_logits=clean_logits.detach(),
            eps_t=eps_t,
            alpha_t=alpha_t,
            steps=int(attack_steps),
            random_start=bool(attack_random_start),
            mean=mean,
            std=std,
        )

        with torch.no_grad():
            adv_features, adv_logits = forward_with_features(model=model, x=x_adv, require_features=True)
            adv_probs = F.softmax(adv_logits, dim=1)
            one_hot_adv = F.one_hot(y_hat, num_classes=adv_probs.size(1)).to(dtype=adv_probs.dtype)
            a_adv = adv_probs - one_hot_adv

        clean_logits_all.append(clean_logits.detach().to(dtype=torch.float32).cpu())
        adv_logits_all.append(adv_logits.detach().to(dtype=torch.float32).cpu())
        clean_features_all.append(clean_features.detach().to(dtype=torch.float32).cpu())
        adv_features_all.append(adv_features.detach().to(dtype=torch.float32).cpu())
        a_clean_all.append(a_clean.detach().to(dtype=torch.float32).cpu())
        a_adv_all.append(a_adv.detach().to(dtype=torch.float32).cpu())
        y_hat_all.append(y_hat.detach().cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=progress_method_name,
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    if was_training:
        model.train()

    if len(clean_logits_all) == 0:
        empty2 = torch.empty((0, 0), dtype=torch.float32)
        empty1 = torch.empty((0,), dtype=torch.long)
        return {
            "clean_logits": empty2,
            "adv_logits": empty2,
            "clean_features": empty2,
            "adv_features": empty2,
            "a_clean": empty2,
            "a_adv": empty2,
            "pseudo_labels": empty1,
        }

    out = {
        "clean_logits": torch.cat(clean_logits_all, dim=0),
        "adv_logits": torch.cat(adv_logits_all, dim=0),
        "clean_features": torch.cat(clean_features_all, dim=0),
        "adv_features": torch.cat(adv_features_all, dim=0),
        "a_clean": torch.cat(a_clean_all, dim=0),
        "a_adv": torch.cat(a_adv_all, dim=0),
        "pseudo_labels": torch.cat(y_hat_all, dim=0),
    }
    for key, value in out.items():
        if value.is_floating_point():
            _assert_finite_tensor(key, value)
    return out


def build_adv_gradient_displacement_explicit_embeddings(components: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Materialize Gamma(x)=g_adv(x)-g_clean(x) as explicit [N, C*d] vectors.

    This is useful for tiny models and debugging, but the kernel path below is
    preferred for large C*d.
    """
    a_clean = components["a_clean"].to(dtype=torch.float32)
    a_adv = components["a_adv"].to(dtype=torch.float32)
    h_clean = components["clean_features"].to(dtype=torch.float32)
    h_adv = components["adv_features"].to(dtype=torch.float32)
    g_clean = a_clean.unsqueeze(2) * h_clean.unsqueeze(1)
    g_adv = a_adv.unsqueeze(2) * h_adv.unsqueeze(1)
    gamma = (g_adv - g_clean).reshape(a_clean.size(0), -1).contiguous()
    _assert_finite_tensor("adv_gradient_displacement_embedding", gamma)
    return gamma


def build_adv_gradient_displacement_gram(
    components: Dict[str, torch.Tensor],
    device: torch.device,
    score_chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Build K_ij=<Gamma_i,Gamma_j> without explicit C*d vectors.

    With g=vec(a h^T), the Kronecker identity gives

        <g_1,g_2> = <a_1,a_2> * <h_1,h_2>.

    Therefore

        <Gamma_i,Gamma_j> =
            <g_adv_i,g_adv_j> + <g_clean_i,g_clean_j>
          - <g_adv_i,g_clean_j> - <g_clean_i,g_adv_j>.

    The downstream objective uses the dual identity

        log det(lambda I + G G^T)
        = C*d*log(lambda) + log det(I_b + K_B/lambda).
    """
    a_clean = components["a_clean"].to(device=device, dtype=torch.float32)
    a_adv = components["a_adv"].to(device=device, dtype=torch.float32)
    h_clean = components["clean_features"].to(device=device, dtype=torch.float32)
    h_adv = components["adv_features"].to(device=device, dtype=torch.float32)

    n = int(a_clean.size(0))
    chunk = n if score_chunk_size <= 0 else max(1, int(score_chunk_size))
    gram = torch.empty((n, n), dtype=torch.float32, device=device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        gram_chunk = (a_adv[start:end] @ a_adv.t()) * (h_adv[start:end] @ h_adv.t())
        gram_chunk += (a_clean[start:end] @ a_clean.t()) * (h_clean[start:end] @ h_clean.t())
        gram_chunk -= (a_adv[start:end] @ a_clean.t()) * (h_adv[start:end] @ h_clean.t())
        gram_chunk -= (a_clean[start:end] @ a_adv.t()) * (h_clean[start:end] @ h_adv.t())
        gram[start:end] = gram_chunk

    gram = 0.5 * (gram + gram.t())
    _assert_finite_tensor("adv_gradient_displacement_gram", gram)
    return gram


def _kernel_quadratic_scores(
    kernel: torch.Tensor,
    diag: torch.Tensor,
    selected: torch.Tensor,
    a_inv: Optional[torch.Tensor],
    lambda_reg: float,
    score_chunk_size: int,
) -> torch.Tensor:
    n = int(kernel.size(0))
    scores = torch.empty((n,), dtype=kernel.dtype, device=kernel.device)
    chunk = n if score_chunk_size <= 0 else max(1, int(score_chunk_size))
    inv = a_inv
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        schur = 1.0 + diag[start:end] / float(lambda_reg)
        if selected.numel() > 0:
            k_chunk = kernel[start:end, selected] / float(lambda_reg)
            quad = (k_chunk @ inv * k_chunk).sum(dim=1)
            schur = schur - quad
        scores[start:end] = schur - 1.0
    return scores


def greedy_logdet_selector_from_gram(
    kernel: torch.Tensor,
    query_size: int,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    ambient_dim: Optional[int] = None,
    progress_logger=None,
    progress_method_name: str = "ADV_GRAD_DISP_LOGDET_GREEDY",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Greedy batch-mode maximization from a Gram matrix K_ij=<Gamma_i,Gamma_j>.

    Selection is performed in the dual:
      F(S)=log det(I + K_S/lambda)
    which is equivalent to the primal last-layer objective up to the constant
      C*d*log(lambda).
    """
    if kernel.ndim != 2 or kernel.size(0) != kernel.size(1):
        raise ValueError(f"kernel must be square [N,N], got shape={tuple(kernel.shape)}")
    if float(lambda_reg) <= 0.0:
        raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
    if float(jitter) <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    k_mat = 0.5 * (kernel.to(dtype=torch.float32) + kernel.to(dtype=torch.float32).t())
    _assert_finite_tensor("kernel", k_mat)
    n = int(k_mat.size(0))
    budget = min(int(query_size), n)
    if budget <= 0 or n == 0:
        return np.array([], dtype=np.int64), {
            "selected_scores": [],
            "selected_log_marginal_gains": [],
            "nonfinite_score_steps": 0,
            "inverse_rebuilds": 0,
            "used_pinv_rebuilds": 0,
            "max_jitter_used": 0.0,
            "initial_logdet_objective": float((ambient_dim or 0) * math.log(float(lambda_reg))),
            "final_logdet_objective": float((ambient_dim or 0) * math.log(float(lambda_reg))),
            "dual_final_logdet_objective": 0.0,
        }

    diag = torch.diag(k_mat).clamp_min(0.0)
    selected_mask = torch.zeros((n,), dtype=torch.bool, device=k_mat.device)
    selected = torch.empty((0,), dtype=torch.long, device=k_mat.device)
    selected_list: List[int] = []
    selected_scores: List[float] = []
    selected_log_marginal_gains: List[float] = []
    nonfinite_score_steps = 0
    inverse_rebuilds = 0
    used_pinv_rebuilds = 0
    max_jitter_used = 0.0

    a_inv: Optional[torch.Tensor] = None
    for step in range(budget):
        scores = _kernel_quadratic_scores(
            kernel=k_mat,
            diag=diag,
            selected=selected,
            a_inv=a_inv,
            lambda_reg=float(lambda_reg),
            score_chunk_size=int(score_chunk_size),
        )
        finite_scores = torch.isfinite(scores)
        if not bool(torch.all(finite_scores)):
            nonfinite_score_steps += 1
            scores = torch.where(finite_scores, scores, torch.full_like(scores, -torch.inf))
        scores = scores.masked_fill(selected_mask, -torch.inf)
        best_local = int(torch.argmax(scores).item())
        best_score = float(scores[best_local].item())
        if not math.isfinite(best_score):
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
            if remaining.numel() == 0:
                break
            best_local = int(remaining[0].item())
            best_score = float(scores[best_local].item()) if torch.isfinite(scores[best_local]) else 0.0

        schur = max(1.0 + best_score, float(jitter))
        selected_mask[best_local] = True
        selected_list.append(best_local)
        selected = torch.as_tensor(selected_list, dtype=torch.long, device=k_mat.device)
        selected_scores.append(float(best_score))
        selected_log_marginal_gains.append(float(math.log(schur)))

        a = torch.eye(selected.numel(), dtype=k_mat.dtype, device=k_mat.device)
        a = a + k_mat[selected][:, selected] / float(lambda_reg)
        a_inv, used_jitter, used_pinv = _inverse_spd_with_jitter(
            matrix=a,
            base_jitter=float(jitter),
            max_tries=6,
        )
        inverse_rebuilds += 1
        max_jitter_used = max(max_jitter_used, float(used_jitter))
        used_pinv_rebuilds += int(used_pinv)

        if progress_logger is not None and ((step + 1) % 10 == 0 or (step + 1) == budget):
            progress_logger.log(
                (
                    f"[{progress_method_name}] step={step + 1}/{budget} "
                    f"best_score={best_score:.6f} rebuilds={inverse_rebuilds}"
                ),
                device=str(k_mat.device),
            )

    selected_np = np.asarray(selected_list, dtype=np.int64)
    if len(selected_np) > 0:
        selected_t = torch.as_tensor(selected_np, dtype=torch.long, device=k_mat.device)
        dual_matrix = torch.eye(len(selected_np), dtype=k_mat.dtype, device=k_mat.device)
        dual_matrix = dual_matrix + k_mat[selected_t][:, selected_t] / float(lambda_reg)
        dual_obj, logdet_jitter, logdet_fallback = _logdet_spd(
            matrix=dual_matrix,
            base_jitter=float(jitter),
            max_tries=6,
        )
    else:
        dual_obj = 0.0
        logdet_jitter = 0.0
        logdet_fallback = False

    primal_const = float((ambient_dim or 0) * math.log(float(lambda_reg)))
    return selected_np, {
        "selected_scores": [float(x) for x in selected_scores],
        "selected_log_marginal_gains": [float(x) for x in selected_log_marginal_gains],
        "initial_logdet_objective": float(primal_const),
        "final_logdet_objective": float(primal_const + float(dual_obj)),
        "dual_final_logdet_objective": float(dual_obj),
        "nonfinite_score_steps": int(nonfinite_score_steps),
        "inverse_rebuilds": int(inverse_rebuilds),
        "used_pinv_rebuilds": int(used_pinv_rebuilds),
        "max_jitter_used": float(max(max_jitter_used, float(logdet_jitter))),
        "dual_logdet_jitter": float(logdet_jitter),
        "dual_logdet_fallback": bool(logdet_fallback),
    }


class LogDetBaseStrategy(BaseAcquisition):
    method_name: str = "logdet_base"
    enable_swap_refinement: bool = False
    embedding_name: str = "logits"
    objective_name: str = "logdet_lambdaI_plus_sum_adv_displacements"
    debug_file_tag: str = "logdet_scores"
    log_prefix: str = "LOGDET"
    displacement_mode: str = "displacement_norm"  # displacement_norm | predictive_ce
    semantic_embedding_space: str = "logits"  # features | logits

    def _compute_displacements(
        self,
        model,
        unlabeled_loader,
        device: torch.device,
        progress_logger,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.displacement_mode == "displacement_norm":
            return compute_adv_displacement_embeddings(
                model=model,
                unlabeled_loader=unlabeled_loader,
                attack_type=self.cfg.logdet_adv_disp_attack,
                attack_norm=self.cfg.logdet_adv_disp_attack_norm,
                epsilon=self.cfg.logdet_adv_disp_epsilon,
                pgd_steps=self.cfg.logdet_adv_disp_pgd_steps,
                pgd_step_size=self.cfg.logdet_adv_disp_pgd_step_size,
                pgd_random_start=self.cfg.logdet_adv_disp_pgd_random_start,
                mean=self.cfg.cifar10_mean,
                std=self.cfg.cifar10_std,
                device=device,
                progress_logger=progress_logger,
                progress_method_name=f"{self.log_prefix}_ATTACK",
                tensor_batch_size=self.cfg.pool_batch_size,
                return_clean_logits=True,
            )
        if self.displacement_mode == "predictive_ce":
            return compute_adv_semantic_displacement_embeddings(
                model=model,
                unlabeled_loader=unlabeled_loader,
                embedding_space=self.semantic_embedding_space,
                attack_type=self.cfg.logdet_adv_disp_attack,
                attack_norm=self.cfg.logdet_adv_disp_attack_norm,
                epsilon=self.cfg.logdet_adv_disp_epsilon,
                pgd_steps=self.cfg.logdet_adv_disp_pgd_steps,
                pgd_step_size=self.cfg.logdet_adv_disp_pgd_step_size,
                pgd_random_start=self.cfg.logdet_adv_disp_pgd_random_start,
                mean=self.cfg.cifar10_mean,
                std=self.cfg.cifar10_std,
                device=device,
                progress_logger=progress_logger,
                progress_method_name=f"{self.log_prefix}_ATTACK",
                tensor_batch_size=self.cfg.pool_batch_size,
                return_clean_logits=True,
            )
        raise ValueError(f"Unsupported displacement_mode: {self.displacement_mode}")

    def _run_logdet_optimizer(
        self,
        displacements: torch.Tensor,
        budget: int,
        progress_logger=None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], float, float, float]:
        """
        Run the repository's original logdet batch optimizer on the provided
        candidate displacement matrix. Callers are responsible for mapping the
        returned local rows back to their original unlabeled-pool indices.
        """
        t_sel = time.perf_counter()
        picked_local, select_debug = greedy_logdet_selector(
            displacements=displacements,
            query_size=budget,
            lambda_reg=self.cfg.logdet_adv_disp_lambda,
            score_chunk_size=self.cfg.logdet_adv_disp_score_chunk_size,
            jitter=self.cfg.logdet_adv_disp_jitter,
            progress_logger=progress_logger,
            progress_method_name=f"{self.log_prefix}_GREEDY",
        )
        greedy_selection_time = time.perf_counter() - t_sel

        swap_debug: Dict[str, Any] = {}
        if bool(self.enable_swap_refinement):
            t_swap = time.perf_counter()
            picked_local, swap_debug = refine_logdet_swaps(
                displacements=displacements,
                selected_local=picked_local,
                lambda_reg=float(self.cfg.logdet_adv_disp_lambda),
                score_chunk_size=int(self.cfg.logdet_adv_disp_score_chunk_size),
                jitter=float(getattr(self.cfg, "logdet_adv_disp_swap_jitter", self.cfg.logdet_adv_disp_jitter)),
                max_swap_rounds=int(getattr(self.cfg, "logdet_adv_disp_swap_max_rounds", 3)),
                swap_top_unselected=int(getattr(self.cfg, "logdet_adv_disp_swap_top_unselected", 200)),
                swap_top_selected=int(getattr(self.cfg, "logdet_adv_disp_swap_top_selected", 0)),
                swap_improvement_tol=float(getattr(self.cfg, "logdet_adv_disp_swap_improvement_tol", 1e-8)),
                swap_downdate_tol=float(getattr(self.cfg, "logdet_adv_disp_swap_downdate_tol", 1e-6)),
                progress_logger=progress_logger,
                progress_method_name=f"{self.log_prefix}_SWAP",
            )
            swap_time = time.perf_counter() - t_swap
        else:
            swap_time = 0.0

        selection_time = float(greedy_selection_time + swap_time)
        return picked_local, select_debug, swap_debug, selection_time, float(greedy_selection_time), float(swap_time)

    def _select_from_cached_embeddings(
        self,
        displacements: torch.Tensor,
        clean_logits_all: torch.Tensor,
        unlabeled_indices: np.ndarray,
        budget: int,
        scoring_time: float,
        device: torch.device,
        progress_logger=None,
    ) -> AcquisitionOutput:
        if displacements.numel() > 0:
            first_step_scores_all = displacements.pow(2).sum(dim=1) / float(self.cfg.logdet_adv_disp_lambda)
            disp_norm_sq_all = displacements.pow(2).sum(dim=1)
            probs = torch.softmax(clean_logits_all, dim=1)
            entropy_scores_all = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        else:
            first_step_scores_all = torch.empty((0,), dtype=torch.float32, device=device)
            disp_norm_sq_all = first_step_scores_all
            entropy_scores_all = first_step_scores_all

        percentile = float(getattr(self.cfg, "logdet_adv_disp_percentile", 0.0))
        if displacements.numel() > 0 and percentile > 0.0:
            threshold = torch.quantile(entropy_scores_all, q=percentile)
            feasible_mask = entropy_scores_all >= threshold
        else:
            threshold = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
            feasible_mask = torch.ones_like(first_step_scores_all, dtype=torch.bool)

        feasible_local = torch.nonzero(feasible_mask, as_tuple=False).squeeze(1)
        if feasible_local.numel() == 0:
            feasible_local = torch.arange(displacements.size(0), device=displacements.device, dtype=torch.long)
            feasible_mask = torch.ones_like(first_step_scores_all, dtype=torch.bool)

        displacements_sel = displacements[feasible_local]

        picked_local, select_debug, swap_debug, selection_time, greedy_selection_time, swap_time = self._run_logdet_optimizer(
            displacements=displacements_sel,
            progress_logger=progress_logger,
            budget=budget,
        )

        picked_local_t = torch.as_tensor(picked_local, device=feasible_local.device, dtype=torch.long)
        picked_local_global = feasible_local[picked_local_t].detach().cpu().numpy()
        selected = unlabeled_indices[picked_local_global]

        first_step_scores = first_step_scores_all
        entropy_scores = entropy_scores_all
        disp_norm_sq = disp_norm_sq_all
        disp_norm_sq_stats = tensor_stats(disp_norm_sq)
        first_step_score_stats = tensor_stats(first_step_scores)
        entropy_score_stats = tensor_stats(entropy_scores)

        k = len(picked_local_global)
        selected_scores = select_debug["selected_scores"]
        selected_mean_score = float(np.mean(selected_scores)) if k > 0 else float("nan")

        debug_data = None
        if bool(getattr(self.cfg, "debug_save_adv_scores", False)):
            selected_flag = np.zeros(len(unlabeled_indices), dtype=np.int64)
            selected_flag[picked_local_global] = 1
            debug_data = {
                "__file_tag": self.debug_file_tag,
                "__column_order": [
                    "index",
                    "disp_norm_sq",
                    "first_step_score",
                    "entropy_score",
                    "feasible_flag",
                    "selected_flag",
                ],
                "index": np.asarray(unlabeled_indices, dtype=np.int64),
                "disp_norm_sq": disp_norm_sq.detach().cpu().numpy(),
                "first_step_score": first_step_scores.detach().cpu().numpy(),
                "entropy_score": entropy_scores.detach().cpu().numpy(),
                "feasible_flag": feasible_mask.detach().cpu().numpy().astype(np.int64),
                "selected_flag": selected_flag,
            }

        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[{self.log_prefix}] disp_norm_sq_stats "
                    f"min={disp_norm_sq_stats['min']:.6f} max={disp_norm_sq_stats['max']:.6f} "
                    f"mean={disp_norm_sq_stats['mean']:.6f} std={disp_norm_sq_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    f"[{self.log_prefix}] greedy_stats "
                    f"selected_mean_score={selected_mean_score:.6f} "
                    f"percentile={percentile:.3f} feasible={int(feasible_local.numel())}/{int(len(unlabeled_indices))} "
                    f"percentile_basis=entropy "
                    f"rebuilds={select_debug['inverse_rebuilds']} "
                    f"nonfinite_steps={select_debug['nonfinite_score_steps']}"
                ),
                device=str(device),
            )
            if bool(self.enable_swap_refinement):
                progress_logger.log(
                    (
                        f"[{self.log_prefix}] swap_stats "
                        f"accepted={int(swap_debug.get('accepted_swaps', 0))} "
                        f"rounds={int(swap_debug.get('swap_rounds_run', 0))} "
                        f"obj_greedy={float(select_debug.get('final_logdet_objective', float('nan'))):.6f} "
                        f"obj_refined={float(swap_debug.get('final_logdet_objective', float('nan'))):.6f}"
                    ),
                    device=str(device),
                )

        method_name = str(getattr(self.cfg, "acquisition_method", self.method_name)).lower()

        return AcquisitionOutput(
            selected_indices=selected,
            scores=first_step_scores.detach().cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": method_name,
                "embedding": self.embedding_name,
                "objective": self.objective_name,
                "attack_type": self.cfg.logdet_adv_disp_attack,
                "attack_norm": self.cfg.logdet_adv_disp_attack_norm,
                "epsilon": float(self.cfg.logdet_adv_disp_epsilon),
                "lambda_reg": float(self.cfg.logdet_adv_disp_lambda),
                "pgd_steps": int(self.cfg.logdet_adv_disp_pgd_steps),
                "pgd_step_size": self.cfg.logdet_adv_disp_pgd_step_size,
                "pgd_random_start": bool(self.cfg.logdet_adv_disp_pgd_random_start),
                "score_chunk_size": int(self.cfg.logdet_adv_disp_score_chunk_size),
                "jitter": float(self.cfg.logdet_adv_disp_jitter),
                "percentile": float(percentile),
                "percentile_basis": "entropy",
                "percentile_threshold": float(threshold.item()) if torch.isfinite(threshold) else float("-inf"),
                "feasible_size": int(feasible_local.numel()),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
                "disp_norm_sq_stats": disp_norm_sq_stats,
                "first_step_score_stats": first_step_score_stats,
                "entropy_score_stats": entropy_score_stats,
                "selected_mean_greedy_score": selected_mean_score,
                "selected_scores": select_debug["selected_scores"],
                "selected_log_marginal_gains": select_debug["selected_log_marginal_gains"],
                "greedy_initial_logdet_objective": float(select_debug.get("initial_logdet_objective", float("nan"))),
                "greedy_final_logdet_objective": float(select_debug.get("final_logdet_objective", float("nan"))),
                "nonfinite_score_steps": int(select_debug["nonfinite_score_steps"]),
                "inverse_rebuilds": int(select_debug["inverse_rebuilds"]),
                "used_pinv_rebuilds": int(select_debug["used_pinv_rebuilds"]),
                "max_jitter_used": float(select_debug["max_jitter_used"]),
                "swap_refinement_enabled": bool(self.enable_swap_refinement),
                "swap_time_sec": float(swap_time),
                "swap_initial_logdet_objective": float(
                    swap_debug.get("initial_logdet_objective", select_debug.get("final_logdet_objective", float("nan")))
                ),
                "swap_final_logdet_objective": float(
                    swap_debug.get("final_logdet_objective", select_debug.get("final_logdet_objective", float("nan")))
                ),
                "swap_accepted_swaps": int(swap_debug.get("accepted_swaps", 0)),
                "swap_rounds_run": int(swap_debug.get("swap_rounds_run", 0)),
                "swap_best_gains_by_round": swap_debug.get("best_swap_gains_by_round", []),
                "swap_accepted_gains": swap_debug.get("accepted_swap_gains", []),
                "swap_downdate_fallback_rebuilds": int(swap_debug.get("downdate_fallback_rebuilds", 0)),
                "swap_downdate_skips": int(swap_debug.get("downdate_skips", 0)),
                "swap_state_rebuilds": int(swap_debug.get("state_rebuilds", 0)),
                "swap_state_used_pinv_rebuilds": int(swap_debug.get("state_used_pinv_rebuilds", 0)),
                "swap_nonfinite_swap_score_events": int(swap_debug.get("nonfinite_swap_score_events", 0)),
                "swap_max_jitter_used": float(swap_debug.get("max_jitter_used", 0.0)),
                "swap_top_unselected": int(swap_debug.get("swap_top_unselected", 0)),
                "swap_top_selected": int(swap_debug.get("swap_top_selected", 0)),
                "swap_improvement_tol": float(
                    swap_debug.get(
                        "swap_improvement_tol",
                        float(getattr(self.cfg, "logdet_adv_disp_swap_improvement_tol", 1e-8)),
                    )
                ),
                "swap_downdate_tol": float(
                    swap_debug.get(
                        "swap_downdate_tol",
                        float(getattr(self.cfg, "logdet_adv_disp_swap_downdate_tol", 1e-6)),
                    )
                ),
            },
            debug_data=debug_data,
        )

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
        displacements, clean_logits_all = self._compute_displacements(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t0
        return self._select_from_cached_embeddings(
            displacements=displacements,
            clean_logits_all=clean_logits_all,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            scoring_time=float(scoring_time),
            device=device,
            progress_logger=progress_logger,
        )


class AdvQTopKStrategy(BaseAcquisition):
    """
    Select the top samples by adversarial viability score:
      q_t(x) = CE(z(x + delta*(x)), argmax z(x))

    The attack delta*(x) maximizes squared logit displacement under an
    L_infinity acquisition budget.
    """

    method_name: str = "adv_q_topk"

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
        displacements, clean_logits_all = compute_adv_displacement_embeddings(
            model=model,
            unlabeled_loader=unlabeled_loader,
            attack_type=getattr(self.cfg, "adv_q_attack_type", "pgd"),
            attack_norm="linf",
            epsilon=float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            pgd_steps=int(getattr(self.cfg, "adv_q_pgd_steps", 3)),
            pgd_step_size=getattr(self.cfg, "adv_q_pgd_step_size", None),
            pgd_random_start=bool(getattr(self.cfg, "adv_q_pgd_random_start", True)),
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            device=device,
            progress_logger=progress_logger,
            progress_method_name="ADV_Q_TOPK_ATTACK",
            tensor_batch_size=self.cfg.pool_batch_size,
            return_clean_logits=True,
        )
        q_scores = compute_adv_q_scores_from_logit_displacements(displacements, clean_logits_all)
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        k = min(int(budget), int(q_scores.numel()))
        if k > 0:
            top_vals, picked_local_t = torch.topk(q_scores, k=k, largest=True)
            picked_local = picked_local_t.detach().cpu().numpy()
        else:
            top_vals = torch.empty((0,), dtype=torch.float32, device=device)
            picked_local = np.array([], dtype=np.int64)
        selected = unlabeled_indices[picked_local]
        selection_time = time.perf_counter() - t1

        q_stats = tensor_stats(q_scores)
        if progress_logger is not None:
            progress_logger.log(
                (
                    "[ADV_Q_TOPK] q_stats "
                    f"min={q_stats['min']:.6f} max={q_stats['max']:.6f} "
                    f"mean={q_stats['mean']:.6f} std={q_stats['std']:.6f} "
                    f"selected={int(len(selected))}"
                ),
                device=str(device),
            )

        return AcquisitionOutput(
            selected_indices=selected,
            scores=q_scores.detach().cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": self.method_name,
                "objective": "adv_viability_q_topk",
                "q_score_stats": q_stats,
                "selected_mean_q_score": float(top_vals.mean().item()) if k > 0 else float("nan"),
                "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
                "attack_type": str(getattr(self.cfg, "adv_q_attack_type", "pgd")),
                "attack_norm": "linf",
                "pgd_steps": int(getattr(self.cfg, "adv_q_pgd_steps", 3)),
                "pgd_step_size": getattr(self.cfg, "adv_q_pgd_step_size", None),
                "pgd_random_start": bool(getattr(self.cfg, "adv_q_pgd_random_start", True)),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
            },
        )


class AdvQFilterLogDetStrategy(LogDetBaseStrategy):
    """
    Two-stage acquisition:
    First filter unlabeled samples by adversarial viability score q_t(x), then
    apply the original logdet batch optimizer on the retained candidate pool.
    """

    method_name: str = "adv_q_filter_logdet"
    enable_swap_refinement: bool = False
    embedding_name: str = "logits"
    objective_name: str = "adv_q_filter_then_logdet_lambdaI_plus_sum_adv_displacements"
    debug_file_tag: str = "adv_q_filter_logdet_scores"
    log_prefix: str = "ADV_Q_FILTER_LOGDET"
    displacement_mode: str = "displacement_norm"
    semantic_embedding_space: str = "logits"

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

        # Shared pass: this computes delta*(x), Delta(x), clean logits, and then
        # q_t(x) without an extra adversarial forward pass.
        t0 = time.perf_counter()
        displacements, clean_logits_all = compute_adv_displacement_embeddings(
            model=model,
            unlabeled_loader=unlabeled_loader,
            attack_type=getattr(self.cfg, "adv_q_attack_type", "pgd"),
            attack_norm="linf",
            epsilon=float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            pgd_steps=int(getattr(self.cfg, "adv_q_pgd_steps", 3)),
            pgd_step_size=getattr(self.cfg, "adv_q_pgd_step_size", None),
            pgd_random_start=bool(getattr(self.cfg, "adv_q_pgd_random_start", True)),
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            device=device,
            progress_logger=progress_logger,
            progress_method_name=f"{self.log_prefix}_ATTACK",
            tensor_batch_size=self.cfg.pool_batch_size,
            return_clean_logits=True,
        )
        q_scores = compute_adv_q_scores_from_logit_displacements(displacements, clean_logits_all)
        scoring_time = time.perf_counter() - t0

        t_filter = time.perf_counter()
        retain_fraction = float(getattr(self.cfg, "retain_fraction", 0.9))
        retained_local = top_retained_local_by_q(q_scores, retain_fraction=retain_fraction)
        filter_time = time.perf_counter() - t_filter

        displacements_retained = displacements[retained_local]
        picked_in_retained, select_debug, swap_debug, logdet_selection_time, greedy_time, swap_time = self._run_logdet_optimizer(
            displacements=displacements_retained,
            budget=budget,
            progress_logger=progress_logger,
        )
        selection_time = float(filter_time + logdet_selection_time)

        picked_in_retained_t = torch.as_tensor(picked_in_retained, device=retained_local.device, dtype=torch.long)
        picked_local_global_t = retained_local[picked_in_retained_t]
        picked_local_global = picked_local_global_t.detach().cpu().numpy()
        selected = unlabeled_indices[picked_local_global]

        disp_norm = displacements.pow(2).sum(dim=1).sqrt() if displacements.numel() > 0 else q_scores
        retained_q = q_scores[retained_local]
        retained_disp_norm = disp_norm[retained_local] if disp_norm.numel() > 0 else disp_norm
        selected_disp_norm = disp_norm[picked_local_global_t] if picked_local_global_t.numel() > 0 else torch.empty(
            (0,), dtype=torch.float32, device=device
        )
        selected_q = q_scores[picked_local_global_t] if picked_local_global_t.numel() > 0 else torch.empty(
            (0,), dtype=torch.float32, device=device
        )

        q_stats = tensor_stats(q_scores)
        retained_q_stats = tensor_stats(retained_q)
        disp_norm_stats = tensor_stats(disp_norm)

        retained_flag = np.zeros(len(unlabeled_indices), dtype=np.int64)
        retained_flag[retained_local.detach().cpu().numpy()] = 1
        selected_flag = np.zeros(len(unlabeled_indices), dtype=np.int64)
        selected_flag[picked_local_global] = 1

        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[{self.log_prefix}] q_full_stats "
                    f"min={q_stats['min']:.6f} max={q_stats['max']:.6f} "
                    f"mean={q_stats['mean']:.6f} std={q_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    f"[{self.log_prefix}] retained_stats "
                    f"retain_fraction={retain_fraction:.4f} "
                    f"retained={int(retained_local.numel())}/{int(len(unlabeled_indices))} "
                    f"q_mean={retained_q_stats['mean']:.6f} q_std={retained_q_stats['std']:.6f} "
                    f"disp_norm_mean={float(retained_disp_norm.mean().item()) if retained_disp_norm.numel() > 0 else float('nan'):.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    f"[{self.log_prefix}] selected_stats "
                    f"selected={int(len(selected))} "
                    f"selected_q_mean={float(selected_q.mean().item()) if selected_q.numel() > 0 else float('nan'):.6f} "
                    f"selected_disp_norm_mean={float(selected_disp_norm.mean().item()) if selected_disp_norm.numel() > 0 else float('nan'):.6f} "
                    f"logdet_obj={float(select_debug.get('final_logdet_objective', float('nan'))):.6f}"
                ),
                device=str(device),
            )

        debug_data = None
        if bool(getattr(self.cfg, "debug_save_adv_scores", False)):
            debug_data = {
                "__file_tag": self.debug_file_tag,
                "__column_order": [
                    "index",
                    "q_score",
                    "disp_norm",
                    "retained_flag",
                    "selected_flag",
                ],
                "index": np.asarray(unlabeled_indices, dtype=np.int64),
                "q_score": q_scores.detach().cpu().numpy(),
                "disp_norm": disp_norm.detach().cpu().numpy(),
                "retained_flag": retained_flag,
                "selected_flag": selected_flag,
            }

        total_time = time.perf_counter() - total_t0
        selected_mean_disp_norm = float(selected_disp_norm.mean().item()) if selected_disp_norm.numel() > 0 else float("nan")
        retained_mean_disp_norm = float(retained_disp_norm.mean().item()) if retained_disp_norm.numel() > 0 else float("nan")

        return AcquisitionOutput(
            selected_indices=selected,
            scores=q_scores.detach().cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": self.method_name,
                "embedding": self.embedding_name,
                "objective": self.objective_name,
                "pipeline": "adv_q_filter_then_original_logdet",
                "retain_fraction": float(retain_fraction),
                "retained_pool_size": int(retained_local.numel()),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
                "q_score_stats": q_stats,
                "retained_q_score_stats": retained_q_stats,
                "disp_norm_stats": disp_norm_stats,
                "retained_mean_disp_norm": retained_mean_disp_norm,
                "selected_mean_disp_norm": selected_mean_disp_norm,
                "mean_grad_disp_score": float(disp_norm.pow(2).mean().item()) if disp_norm.numel() > 0 else float("nan"),
                "selected_mean_grad_disp_score": (
                    float(selected_disp_norm.pow(2).mean().item()) if selected_disp_norm.numel() > 0 else float("nan")
                ),
                "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
                "attack_type": str(getattr(self.cfg, "adv_q_attack_type", "pgd")),
                "attack_norm": "linf",
                "pgd_steps": int(getattr(self.cfg, "adv_q_pgd_steps", 3)),
                "pgd_step_size": getattr(self.cfg, "adv_q_pgd_step_size", None),
                "pgd_random_start": bool(getattr(self.cfg, "adv_q_pgd_random_start", True)),
                "lambda_reg": float(self.cfg.logdet_adv_disp_lambda),
                "score_chunk_size": int(self.cfg.logdet_adv_disp_score_chunk_size),
                "jitter": float(self.cfg.logdet_adv_disp_jitter),
                "selected_scores": select_debug["selected_scores"],
                "selected_log_marginal_gains": select_debug["selected_log_marginal_gains"],
                "greedy_initial_logdet_objective": float(select_debug.get("initial_logdet_objective", float("nan"))),
                "greedy_final_logdet_objective": float(select_debug.get("final_logdet_objective", float("nan"))),
                "final_logdet_objective": float(select_debug.get("final_logdet_objective", float("nan"))),
                "nonfinite_score_steps": int(select_debug["nonfinite_score_steps"]),
                "inverse_rebuilds": int(select_debug["inverse_rebuilds"]),
                "used_pinv_rebuilds": int(select_debug["used_pinv_rebuilds"]),
                "max_jitter_used": float(select_debug["max_jitter_used"]),
                "timing_q_scores_sec": float(scoring_time),
                "timing_filter_sec": float(filter_time),
                "timing_greedy_selection_sec": float(greedy_time),
                "timing_logdet_selection_sec": float(logdet_selection_time),
                "timing_total_acquisition_sec": float(total_time),
                "swap_refinement_enabled": bool(self.enable_swap_refinement),
                "swap_time_sec": float(swap_time),
                "swap_final_logdet_objective": float(
                    swap_debug.get("final_logdet_objective", select_debug.get("final_logdet_objective", float("nan")))
                ),
            },
            debug_data=debug_data,
        )


class LogDetAdvDispStrategy(LogDetBaseStrategy):
    method_name: str = "logdet_adv_disp"
    enable_swap_refinement: bool = False
    embedding_name: str = "logits"
    objective_name: str = "logdet_lambdaI_plus_sum_adv_displacements"
    debug_file_tag: str = "logdet_adv_disp_scores"
    log_prefix: str = "LOGDET_ADV_DISP"
    displacement_mode: str = "displacement_norm"
    semantic_embedding_space: str = "logits"


class LogDetAdvDispSwapStrategy(LogDetAdvDispStrategy):
    method_name: str = "logdet_adv_disp_swap"
    enable_swap_refinement: bool = True


class LogDetAdvFeatSwapStrategy(LogDetBaseStrategy):
    method_name: str = "logdet_adv_feat_swap"
    enable_swap_refinement: bool = True
    embedding_name: str = "features"
    objective_name: str = "logdet_lambdaI_plus_sum_adv_feature_displacements_predce"
    debug_file_tag: str = "logdet_adv_feat_swap_scores"
    log_prefix: str = "LOGDET_ADV_FEAT_SWAP"
    displacement_mode: str = "predictive_ce"
    semantic_embedding_space: str = "features"


class LogDetAdvLogitSwapStrategy(LogDetBaseStrategy):
    method_name: str = "logdet_adv_logit_swap"
    enable_swap_refinement: bool = True
    embedding_name: str = "logits"
    objective_name: str = "logdet_lambdaI_plus_sum_adv_logit_displacements_predce"
    debug_file_tag: str = "logdet_adv_logit_swap_scores"
    log_prefix: str = "LOGDET_ADV_LOGIT_SWAP"
    displacement_mode: str = "predictive_ce"
    semantic_embedding_space: str = "logits"


class AdvGradDisplacementLogDetStrategy(BaseAcquisition):
    """
    BADGE-inspired adversarial gradient-displacement logdet acquisition.

    For each unlabeled x:
      g_clean = vec((p_clean - e_yhat) h_clean^T)
      g_adv   = vec((p_adv   - e_yhat) h_adv^T)
      Gamma   = g_adv - g_clean

    The selected batch maximizes log det(lambda I + sum Gamma Gamma^T).
    By default we avoid explicit C*d vectors and optimize the dual objective
    with the Gram matrix K_ij=<Gamma_i,Gamma_j>.
    """

    method_name: str = "adv_grad_displacement_logdet"
    log_prefix: str = "ADV_GRAD_DISP_LOGDET"
    debug_file_tag: str = "adv_grad_displacement_logdet_scores"

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
        attack_steps = int(getattr(self.cfg, "attack_steps", 3))
        attack_step_size = getattr(self.cfg, "attack_step_size", None)
        attack_random_start = bool(getattr(self.cfg, "attack_random_start", True))
        eps_acq = float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0))
        lambda_reg = float(getattr(self.cfg, "logdet_lambda", getattr(self.cfg, "logdet_adv_disp_lambda", 1e-3)))
        score_chunk_size = int(getattr(self.cfg, "logdet_adv_disp_score_chunk_size", 8192))
        jitter = float(getattr(self.cfg, "logdet_adv_disp_jitter", 1e-8))
        use_explicit = bool(getattr(self.cfg, "adv_grad_displacement_use_explicit_embedding", False))

        t_score = time.perf_counter()
        components = compute_adv_gradient_displacement_components(
            model=model,
            unlabeled_loader=unlabeled_loader,
            epsilon=eps_acq,
            attack_steps=attack_steps,
            attack_step_size=attack_step_size,
            attack_random_start=attack_random_start,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            device=device,
            progress_logger=progress_logger,
            progress_method_name=f"{self.log_prefix}_ATTACK",
            tensor_batch_size=self.cfg.pool_batch_size,
        )
        clean_logits = components["clean_logits"]
        adv_logits = components["adv_logits"]
        clean_features = components["clean_features"]
        adv_features = components["adv_features"]
        n = int(clean_logits.size(0))
        c = int(components["a_clean"].size(1)) if n > 0 else 0
        d = int(clean_features.size(1)) if n > 0 else 0
        ambient_dim = int(c * d)

        z_disp_norm = (adv_logits - clean_logits).norm(dim=1) if n > 0 else torch.empty((0,), dtype=torch.float32)
        h_disp_norm = (adv_features - clean_features).norm(dim=1) if n > 0 else torch.empty((0,), dtype=torch.float32)
        z_disp_stats = tensor_stats(z_disp_norm)
        h_disp_stats = tensor_stats(h_disp_norm)

        if use_explicit:
            embeddings = build_adv_gradient_displacement_explicit_embeddings(components).to(device=device)
            gamma_norm_sq = embeddings.pow(2).sum(dim=1).detach().cpu()
            scoring_time = time.perf_counter() - t_score
            t_select = time.perf_counter()
            picked_local, select_debug = greedy_logdet_selector(
                displacements=embeddings,
                query_size=budget,
                lambda_reg=lambda_reg,
                score_chunk_size=score_chunk_size,
                jitter=jitter,
                progress_logger=progress_logger,
                progress_method_name=f"{self.log_prefix}_EXPLICIT_GREEDY",
            )
            selection_time = time.perf_counter() - t_select
            selector_mode = "explicit_embedding"
            if len(picked_local) > 0:
                picked_t = torch.as_tensor(picked_local, dtype=torch.long, device=embeddings.device)
                selected_kernel = embeddings[picked_t] @ embeddings[picked_t].t()
            else:
                selected_kernel = torch.empty((0, 0), dtype=torch.float32, device=device)
            gram_symmetry_max_abs = 0.0
            selection_validation = {
                "gram_symmetric": True,
                "gram_symmetry_max_abs": 0.0,
                "kernel_contains_finite": True,
            }
        else:
            kernel = build_adv_gradient_displacement_gram(
                components=components,
                device=device,
                score_chunk_size=score_chunk_size,
            )
            gram_symmetry_max_abs = float((kernel - kernel.t()).abs().max().item()) if kernel.numel() > 0 else 0.0
            if gram_symmetry_max_abs > 1e-4:
                raise ValueError(f"adv gradient displacement Gram is not symmetric: max_abs={gram_symmetry_max_abs}")
            gamma_norm_sq = torch.diag(kernel).clamp_min(0.0).detach().cpu()
            scoring_time = time.perf_counter() - t_score
            t_select = time.perf_counter()
            picked_local, select_debug = greedy_logdet_selector_from_gram(
                kernel=kernel,
                query_size=budget,
                lambda_reg=lambda_reg,
                score_chunk_size=score_chunk_size,
                jitter=jitter,
                ambient_dim=ambient_dim,
                progress_logger=progress_logger,
                progress_method_name=f"{self.log_prefix}_GRAM_GREEDY",
            )
            selection_time = time.perf_counter() - t_select
            selector_mode = "kernel_trick"
            if len(picked_local) > 0:
                picked_t = torch.as_tensor(picked_local, dtype=torch.long, device=kernel.device)
                selected_kernel = kernel[picked_t][:, picked_t]
            else:
                selected_kernel = torch.empty((0, 0), dtype=torch.float32, device=device)
            selection_validation = {
                "gram_symmetric": bool(gram_symmetry_max_abs <= 1e-4),
                "gram_symmetry_max_abs": float(gram_symmetry_max_abs),
                "kernel_contains_finite": bool(torch.isfinite(kernel).all().item()) if kernel.numel() > 0 else True,
            }

        selected = unlabeled_indices[picked_local]
        if len(np.unique(picked_local)) != len(picked_local):
            raise ValueError("adv_grad_displacement_logdet selected duplicate local indices")
        if len(selected) != min(int(budget), int(n)):
            raise ValueError(
                "adv_grad_displacement_logdet returned unexpected number of samples: "
                f"got {len(selected)}, expected {min(int(budget), int(n))}"
            )

        gamma_norm = gamma_norm_sq.clamp_min(0.0).sqrt()
        gamma_norm_stats = tensor_stats(gamma_norm)
        selected_gamma_norm = gamma_norm[torch.as_tensor(picked_local, dtype=torch.long)] if len(picked_local) > 0 else torch.empty((0,))
        selected_gamma_norm_stats = tensor_stats(selected_gamma_norm)

        if selected_kernel.numel() > 0:
            selected_kernel_cpu = selected_kernel.detach().cpu()
            diag_mask = torch.eye(selected_kernel_cpu.size(0), dtype=torch.bool)
            offdiag = selected_kernel_cpu[~diag_mask] if selected_kernel_cpu.numel() > selected_kernel_cpu.size(0) else torch.empty((0,))
            selected_kernel_stats = tensor_stats(selected_kernel_cpu.reshape(-1))
            selected_kernel_offdiag_stats = tensor_stats(offdiag)
        else:
            selected_kernel_stats = tensor_stats(torch.empty((0,)))
            selected_kernel_offdiag_stats = tensor_stats(torch.empty((0,)))

        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[{self.log_prefix}] displacement_stats "
                    f"z_mean={z_disp_stats['mean']:.6f} z_std={z_disp_stats['std']:.6f} "
                    f"h_mean={h_disp_stats['mean']:.6f} h_std={h_disp_stats['std']:.6f} "
                    f"gamma_mean={gamma_norm_stats['mean']:.6f} gamma_std={gamma_norm_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    f"[{self.log_prefix}] selected_stats "
                    f"selected={int(len(selected))} mode={selector_mode} "
                    f"logdet_obj={float(select_debug.get('final_logdet_objective', float('nan'))):.6f} "
                    f"dual_logdet={float(select_debug.get('dual_final_logdet_objective', float('nan'))):.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    f"[{self.log_prefix}] selected_kernel_stats "
                    f"mean={selected_kernel_stats['mean']:.6f} std={selected_kernel_stats['std']:.6f} "
                    f"offdiag_mean={selected_kernel_offdiag_stats['mean']:.6f} "
                    f"offdiag_std={selected_kernel_offdiag_stats['std']:.6f}"
                ),
                device=str(device),
            )

        selected_flag = np.zeros(len(unlabeled_indices), dtype=np.int64)
        selected_flag[picked_local] = 1
        debug_data = None
        if bool(getattr(self.cfg, "debug_save_adv_scores", False)):
            debug_data = {
                "__file_tag": self.debug_file_tag,
                "__column_order": [
                    "index",
                    "z_disp_norm",
                    "h_disp_norm",
                    "gamma_norm",
                    "selected_flag",
                ],
                "index": np.asarray(unlabeled_indices, dtype=np.int64),
                "z_disp_norm": z_disp_norm.detach().cpu().numpy(),
                "h_disp_norm": h_disp_norm.detach().cpu().numpy(),
                "gamma_norm": gamma_norm.detach().cpu().numpy(),
                "selected_flag": selected_flag,
            }

        total_time = time.perf_counter() - total_t0
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=gamma_norm_sq.detach().cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": self.method_name,
                "embedding": "adv_last_layer_gradient_displacement",
                "selector_mode": selector_mode,
                "objective": "logdet_lambdaI_plus_sum_gamma_gammaT",
                "dual_objective": "logdet_I_plus_K_over_lambda",
                "kernel_trick": bool(not use_explicit),
                "explicit_embedding": bool(use_explicit),
                "ambient_dim": int(ambient_dim),
                "num_classes": int(c),
                "feature_dim": int(d),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
                "epsilon_acq": float(eps_acq),
                "attack_norm": "linf",
                "attack_objective": "squared_logit_displacement",
                "attack_steps": int(attack_steps),
                "attack_step_size": attack_step_size,
                "attack_random_start": bool(attack_random_start),
                "lambda_reg": float(lambda_reg),
                "logdet_lambda": float(lambda_reg),
                "score_chunk_size": int(score_chunk_size),
                "jitter": float(jitter),
                "z_disp_norm_stats": z_disp_stats,
                "h_disp_norm_stats": h_disp_stats,
                "gamma_norm_stats": gamma_norm_stats,
                "selected_gamma_norm_stats": selected_gamma_norm_stats,
                "selected_kernel_stats": selected_kernel_stats,
                "selected_kernel_offdiag_stats": selected_kernel_offdiag_stats,
                "mean_grad_disp_score": float(gamma_norm_sq.mean().item()) if gamma_norm_sq.numel() > 0 else float("nan"),
                "selected_mean_grad_disp_score": (
                    float(selected_gamma_norm.pow(2).mean().item()) if selected_gamma_norm.numel() > 0 else float("nan")
                ),
                "selected_scores": select_debug["selected_scores"],
                "selected_log_marginal_gains": select_debug["selected_log_marginal_gains"],
                "greedy_initial_logdet_objective": float(select_debug.get("initial_logdet_objective", float("nan"))),
                "greedy_final_logdet_objective": float(select_debug.get("final_logdet_objective", float("nan"))),
                "dual_final_logdet_objective": float(select_debug.get("dual_final_logdet_objective", float("nan"))),
                "final_logdet_objective": float(select_debug.get("final_logdet_objective", float("nan"))),
                "nonfinite_score_steps": int(select_debug["nonfinite_score_steps"]),
                "inverse_rebuilds": int(select_debug["inverse_rebuilds"]),
                "used_pinv_rebuilds": int(select_debug["used_pinv_rebuilds"]),
                "max_jitter_used": float(select_debug["max_jitter_used"]),
                "validation": {
                    **selection_validation,
                    "selected_unique": bool(len(np.unique(picked_local)) == len(picked_local)),
                    "selected_count_matches_budget": bool(len(selected) == min(int(budget), int(n))),
                },
                "timing_total_acquisition_sec": float(total_time),
                "timing_scoring_sec": float(scoring_time),
                "timing_selection_sec": float(selection_time),
            },
            debug_data=debug_data,
        )

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from acquisition.badge import last_layer_gradient_embedding_from_logits_features
from acquisition.logdet_adv_disp import (
    _fgsm_predictive_ce_attack,
    _pgd_predictive_ce_attack,
    forward_with_features,
)
from acquisition.secant_badge import _secant_attack_settings
from acquisition.utils import AcquisitionOutput, BaseAcquisition, scaled_linf_eps, tensor_stats


def _scalar_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    xf = x.float().detach().cpu()
    return {
        "min": float(xf.min().item()),
        "median": float(torch.median(xf).item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
    }


def compute_corr_residual_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    epsilon_acq: float = 1.0 / 255.0,
    attack_type: str = "pgd",
    attack_steps: int = 3,
    attack_step_size: Optional[float] = None,
    attack_random_start: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    progress_logger=None,
) -> Dict[str, torch.Tensor]:
    """
    Compute clean last-layer pseudo-gradients and correction vectors in batches.

    The pseudo-label is fixed from the clean prediction and reused both for the
    acquisition-time attack and for clean/adversarial last-layer gradients. Bias
    gradients are omitted, matching the repo's BADGE convention.
    """
    if device is None:
        device = next(model.parameters()).device
    if int(attack_steps) <= 0:
        raise ValueError(f"attack_steps must be positive, got {attack_steps}")
    attack_type = str(attack_type).lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    was_training = model.training
    model.eval()

    g_clean_all: List[torch.Tensor] = []
    gamma_all: List[torch.Tensor] = []
    clean_norms: List[torch.Tensor] = []
    adv_norms: List[torch.Tensor] = []
    gamma_norms: List[torch.Tensor] = []

    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        x0 = images.to(device, non_blocking=True).detach()
        with torch.no_grad():
            clean_features, clean_logits = forward_with_features(model=model, x=x0, require_features=True)
            clean_probs = torch.softmax(clean_logits, dim=1)
            pseudo_labels = clean_probs.argmax(dim=1)
            g_clean = last_layer_gradient_embedding_from_logits_features(
                logits=clean_logits,
                features=clean_features,
                pseudo_labels=pseudo_labels,
            )

        if float(epsilon_acq) <= 0.0:
            adv_features = clean_features
            adv_logits = clean_logits
        else:
            channels = int(x0.size(1))
            eps_t = scaled_linf_eps(
                epsilon=float(epsilon_acq),
                std=std,
                device=x0.device,
                dtype=x0.dtype,
                channels=channels,
            )
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
                step_size = (
                    float(epsilon_acq) / max(float(attack_steps) / 2.0, 1.0)
                    if attack_step_size is None
                    else float(attack_step_size)
                )
                if step_size <= 0.0:
                    raise ValueError(f"attack_step_size must be positive or None, got {attack_step_size}")
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
                    steps=int(attack_steps),
                    random_start=bool(attack_random_start),
                    mean=mean,
                    std=std,
                )
            with torch.no_grad():
                adv_features, adv_logits = forward_with_features(model=model, x=x_adv, require_features=True)

        with torch.no_grad():
            g_adv = last_layer_gradient_embedding_from_logits_features(
                logits=adv_logits,
                features=adv_features,
                pseudo_labels=pseudo_labels,
            )
            gamma = g_adv - g_clean
            g_clean_all.append(g_clean.detach().to(dtype=torch.float32, device="cpu"))
            gamma_all.append(gamma.detach().to(dtype=torch.float32, device="cpu"))
            clean_norms.append(g_clean.float().norm(dim=1).detach().cpu())
            adv_norms.append(g_adv.float().norm(dim=1).detach().cpu())
            gamma_norms.append(gamma.float().norm(dim=1).detach().cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="OURS_CORR_RESIDUAL_REFINE",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    if was_training:
        model.train()

    if len(g_clean_all) == 0:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return {
            "g_clean": empty,
            "gamma": empty,
            "clean_norm": torch.empty((0,), dtype=torch.float32),
            "adv_norm": torch.empty((0,), dtype=torch.float32),
            "gamma_norm": torch.empty((0,), dtype=torch.float32),
        }

    return {
        "g_clean": torch.cat(g_clean_all, dim=0),
        "gamma": torch.cat(gamma_all, dim=0),
        "clean_norm": torch.cat(clean_norms, dim=0),
        "adv_norm": torch.cat(adv_norms, dim=0),
        "gamma_norm": torch.cat(gamma_norms, dim=0),
    }


def clean_gradient_percentile_gate(
    clean_norms: torch.Tensor,
    drop_percentile: float,
    budget: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Hard clean-gradient gate: drop the bottom p% by ||g_clean||.

    The threshold tau is derived from the current pool scores. We drop
    floor(p*N/100) samples unless that would leave fewer than budget samples.
    Ties at tau are retained by the hard >= tau rule.
    """
    if clean_norms.ndim != 1:
        raise ValueError(f"clean_norms must be [N], got shape={tuple(clean_norms.shape)}")
    if not (0.0 <= float(drop_percentile) <= 100.0):
        raise ValueError(f"clean_gate_percentile must be in [0,100], got {drop_percentile}")

    n = int(clean_norms.numel())
    if n == 0:
        return np.array([], dtype=np.int64), {
            "enabled": False,
            "drop_percentile": float(drop_percentile),
            "tau": float("nan"),
            "pool_size_before": 0,
            "pool_size_after": 0,
            "requested_drop_count": 0,
            "actual_drop_count": 0,
            "score_stats": _scalar_stats(clean_norms),
            "retained_score_stats": _scalar_stats(clean_norms),
        }

    scores_np = clean_norms.detach().cpu().numpy().astype(np.float64, copy=False)
    max_drop = max(0, n - min(max(int(budget), 0), n))
    requested_drop = int(np.floor(float(drop_percentile) * float(n) / 100.0))
    drop_count = min(max(requested_drop, 0), max_drop)
    if float(drop_percentile) <= 0.0 or drop_count <= 0:
        retained = np.arange(n, dtype=np.int64)
        tau = float("-inf")
        enabled = False
    else:
        order = np.lexsort((np.arange(n, dtype=np.int64), scores_np))
        tau = float(scores_np[order[drop_count]])
        retained = np.flatnonzero(scores_np >= tau).astype(np.int64)
        enabled = True

    retained_scores = clean_norms[torch.as_tensor(retained, dtype=torch.long)]
    return retained, {
        "enabled": bool(enabled),
        "drop_percentile": float(drop_percentile),
        "tau": float(tau),
        "pool_size_before": int(n),
        "pool_size_after": int(len(retained)),
        "requested_drop_count": int(requested_drop),
        "actual_drop_count": int(n - len(retained)),
        "score_stats": _scalar_stats(clean_norms),
        "retained_score_stats": _scalar_stats(retained_scores),
    }


def _build_correction_target(gamma: torch.Tensor, target: str = "mean", eps: float = 1e-12) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if gamma.ndim != 2:
        raise ValueError(f"gamma must be [N,D], got shape={tuple(gamma.shape)}")
    target = str(target).lower()
    if target != "mean":
        raise ValueError(f"Unsupported correction target: {target}")
    if gamma.size(0) == 0:
        v = torch.empty((gamma.size(1),), dtype=torch.float64)
    else:
        v = gamma.to(dtype=torch.float64).mean(dim=0)
    raw_norm = float(v.norm().item()) if v.numel() > 0 else 0.0
    if raw_norm > eps:
        v = v / (raw_norm + eps)
    return v, {"target_type": target, "target_raw_norm": raw_norm, "target_normalized": bool(raw_norm > eps)}


def _mgs_basis_rows(vectors: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be [K,D], got shape={tuple(vectors.shape)}")
    d = int(vectors.size(1))
    basis: List[torch.Tensor] = []
    for row in vectors.to(dtype=torch.float64):
        q = row.clone()
        for b in basis:
            q = q - torch.dot(q, b) * b
        for b in basis:
            q = q - torch.dot(q, b) * b
        norm = q.norm()
        if torch.isfinite(norm) and float(norm.item()) > eps:
            basis.append(q / norm)
    if len(basis) == 0:
        return vectors.new_empty((0, d), dtype=torch.float64)
    return torch.stack(basis, dim=0)


def _append_mgs_basis_row(q_rows: torch.Tensor, vector: torch.Tensor, eps: float = 1e-10) -> Tuple[torch.Tensor, bool]:
    q = vector.to(dtype=torch.float64).clone()
    if q_rows.numel() > 0:
        coeff = q_rows @ q
        q = q - coeff @ q_rows
        coeff = q_rows @ q
        q = q - coeff @ q_rows
    norm = q.norm()
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return q_rows, False
    q = (q / norm).view(1, -1)
    if q_rows.numel() == 0:
        return q, True
    return torch.cat([q_rows, q], dim=0), True


def _residual_from_basis(q_rows: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q_rows.numel() == 0:
        return v.clone()
    coeff = q_rows @ v
    return v - coeff @ q_rows


def _projector_objective_from_basis(q_rows: torch.Tensor, v: torch.Tensor) -> float:
    if q_rows.numel() == 0:
        return 0.0
    coeff = q_rows @ v
    return float(torch.dot(coeff, coeff).item())


def _projector_objective_from_gram(
    gram: torch.Tensor,
    target_dot: torch.Tensor,
    subset: Sequence[int],
    eig_tol: float = 1e-10,
) -> float:
    if len(subset) == 0:
        return 0.0
    idx = torch.as_tensor(list(subset), dtype=torch.long)
    g = gram.index_select(0, idx).index_select(1, idx).to(dtype=torch.float64)
    c = target_dot.index_select(0, idx).to(dtype=torch.float64)
    g = 0.5 * (g + g.t())
    try:
        evals, evecs = torch.linalg.eigh(g)
    except RuntimeError:
        return float("-inf")
    max_eval = float(evals.abs().max().item()) if evals.numel() > 0 else 0.0
    keep = evals > max(float(eig_tol), float(eig_tol) * max_eval)
    if not bool(keep.any()):
        return 0.0
    coeff = evecs.t() @ c
    obj = (coeff[keep].pow(2) / evals[keep].clamp_min(eig_tol)).sum()
    if not torch.isfinite(obj):
        return float("-inf")
    return float(max(0.0, obj.item()))


def _batched_forward_scores(
    gamma: torch.Tensor,
    gamma_norms: torch.Tensor,
    residual: torch.Tensor,
    available_mask: torch.Tensor,
    chunk_size: int,
    eps: float,
) -> torch.Tensor:
    n = int(gamma.size(0))
    scores = torch.full((n,), float("-inf"), dtype=torch.float64)
    if n == 0 or not bool(available_mask.any()):
        return scores
    r = residual.to(dtype=gamma.dtype)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        mask = available_mask[start:end]
        if not bool(mask.any()):
            continue
        dots = gamma[start:end] @ r
        denom = gamma_norms[start:end].to(dtype=dots.dtype).clamp_min(float(eps))
        local_scores = torch.abs(dots / denom).to(dtype=torch.float64)
        local_scores = torch.where(mask, local_scores, torch.full_like(local_scores, float("-inf")))
        scores[start:end] = local_scores
    return scores


def _top_clean_fallback(
    clean_norms: torch.Tensor,
    available_mask: torch.Tensor,
    count: int,
) -> List[int]:
    if count <= 0 or not bool(available_mask.any()):
        return []
    available = torch.nonzero(available_mask, as_tuple=False).flatten().cpu().numpy().astype(np.int64)
    scores = clean_norms[torch.as_tensor(available, dtype=torch.long)].cpu().numpy()
    order = np.lexsort((available, -scores))
    return available[order[:count]].astype(np.int64).tolist()


def residual_forward_select(
    gamma: torch.Tensor,
    clean_norms: torch.Tensor,
    v_target: torch.Tensor,
    budget: int,
    score_chunk_size: int = 8192,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    if gamma.ndim != 2:
        raise ValueError(f"gamma must be [N,D], got shape={tuple(gamma.shape)}")
    if clean_norms.ndim != 1 or clean_norms.numel() != gamma.size(0):
        raise ValueError("clean_norms must be [N] and match gamma rows")
    if v_target.ndim != 1 or v_target.numel() != gamma.size(1):
        raise ValueError("v_target must be [D] and match gamma columns")

    n = int(gamma.size(0))
    budget = min(max(int(budget), 0), n)
    selected: List[int] = []
    q_rows = gamma.new_empty((0, gamma.size(1)), dtype=torch.float64)
    available = torch.ones((n,), dtype=torch.bool)
    gamma_norms = gamma.float().norm(dim=1)
    objectives: List[float] = []
    residual_norms: List[float] = []
    selected_scores: List[float] = []
    rank_updates = 0
    fallback_count = 0

    if budget == 0:
        return np.array([], dtype=np.int64), q_rows, {
            "forward_objectives": objectives,
            "residual_norms": residual_norms,
            "selected_forward_scores": selected_scores,
            "basis_rank": 0,
            "rank_updates": 0,
            "fallback_count": 0,
        }

    if float(v_target.norm().item()) <= eps or float(gamma_norms.max().item()) <= eps:
        fallback = _top_clean_fallback(clean_norms=clean_norms, available_mask=available, count=budget)
        return np.asarray(fallback, dtype=np.int64), q_rows, {
            "forward_objectives": [0.0 for _ in fallback],
            "residual_norms": [0.0 for _ in fallback],
            "selected_forward_scores": [0.0 for _ in fallback],
            "basis_rank": 0,
            "rank_updates": 0,
            "fallback_count": len(fallback),
            "fallback_reason": "zero_correction_target_or_atoms",
        }

    for _ in range(budget):
        residual = _residual_from_basis(q_rows, v_target)
        residual_norm = float(residual.norm().item())
        residual_norms.append(residual_norm)
        if residual_norm <= eps:
            picked_more = _top_clean_fallback(clean_norms, available, 1)
            if not picked_more:
                break
            picked = int(picked_more[0])
            best_score = 0.0
            fallback_count += 1
        else:
            scores = _batched_forward_scores(
                gamma=gamma,
                gamma_norms=gamma_norms,
                residual=residual,
                available_mask=available,
                chunk_size=score_chunk_size,
                eps=eps,
            )
            best_score_t, picked_t = torch.max(scores, dim=0)
            best_score = float(best_score_t.item())
            if not np.isfinite(best_score) or best_score <= eps:
                picked_more = _top_clean_fallback(clean_norms, available, 1)
                if not picked_more:
                    break
                picked = int(picked_more[0])
                best_score = 0.0
                fallback_count += 1
            else:
                picked = int(picked_t.item())

        selected.append(picked)
        available[picked] = False
        q_rows, updated = _append_mgs_basis_row(q_rows, gamma[picked], eps=max(eps, 1e-10))
        rank_updates += int(updated)
        selected_scores.append(best_score)
        objectives.append(_projector_objective_from_basis(q_rows, v_target))

    return np.asarray(selected, dtype=np.int64), q_rows, {
        "forward_objectives": objectives,
        "residual_norms": residual_norms,
        "selected_forward_scores": selected_scores,
        "basis_rank": int(q_rows.size(0)),
        "rank_updates": int(rank_updates),
        "fallback_count": int(fallback_count),
    }


def refine_by_residual_swaps(
    gamma: torch.Tensor,
    clean_norms: torch.Tensor,
    selected_local: np.ndarray,
    v_target: torch.Tensor,
    score_chunk_size: int = 8192,
    max_rounds: int = 3,
    incoming_shortlist: int = 64,
    outgoing_shortlist: int = 8,
    improvement_tol: float = 1e-8,
    eig_tol: float = 1e-10,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(gamma.size(0))
    selected = [int(x) for x in np.asarray(selected_local, dtype=np.int64).tolist()]
    if len(selected) == 0 or n == 0 or int(max_rounds) <= 0:
        q_rows = _mgs_basis_rows(gamma[selected] if selected else gamma.new_empty((0, gamma.size(1))), eps=max(eps, 1e-10))
        return np.asarray(selected, dtype=np.int64), {
            "initial_objective": _projector_objective_from_basis(q_rows, v_target),
            "final_objective": _projector_objective_from_basis(q_rows, v_target),
            "accepted_swaps": 0,
            "swap_rounds_run": 0,
            "accepted_gains": [],
            "best_gains_by_round": [],
        }

    selected_set = set(selected)
    accepted_gains: List[float] = []
    best_gains_by_round: List[float] = []
    q_rows = _mgs_basis_rows(gamma[selected], eps=max(eps, 1e-10))
    current_obj = _projector_objective_from_basis(q_rows, v_target)
    initial_obj = current_obj
    gamma_norms = gamma.float().norm(dim=1)

    for round_idx in range(int(max_rounds)):
        q_rows = _mgs_basis_rows(gamma[selected], eps=max(eps, 1e-10))
        current_obj = _projector_objective_from_basis(q_rows, v_target)
        selected_set = set(selected)
        available = torch.ones((n,), dtype=torch.bool)
        if selected:
            available[torch.as_tensor(selected, dtype=torch.long)] = False
        residual = _residual_from_basis(q_rows, v_target)
        incoming_scores = _batched_forward_scores(
            gamma=gamma,
            gamma_norms=gamma_norms,
            residual=residual,
            available_mask=available,
            chunk_size=score_chunk_size,
            eps=eps,
        )
        incoming_candidates = torch.nonzero(available, as_tuple=False).flatten()
        if incoming_candidates.numel() == 0:
            break
        k_in = int(incoming_candidates.numel()) if int(incoming_shortlist) <= 0 else min(int(incoming_shortlist), int(incoming_candidates.numel()))
        if float(residual.norm().item()) <= eps or not torch.isfinite(incoming_scores[incoming_candidates]).any():
            incoming_np = _top_clean_fallback(clean_norms, available, k_in)
        else:
            top_rel = torch.topk(incoming_scores[incoming_candidates], k=k_in, largest=True).indices
            incoming_np = incoming_candidates[top_rel].cpu().numpy().astype(np.int64).tolist()

        contributions: List[Tuple[float, int]] = []
        current_selected_positions = list(selected)
        for pos, member in enumerate(current_selected_positions):
            without = current_selected_positions[:pos] + current_selected_positions[pos + 1 :]
            q_wo = _mgs_basis_rows(gamma[without] if without else gamma.new_empty((0, gamma.size(1))), eps=max(eps, 1e-10))
            obj_wo = _projector_objective_from_basis(q_wo, v_target)
            contributions.append((current_obj - obj_wo, member))
        contributions.sort(key=lambda item: (item[0], item[1]))
        k_out = len(contributions) if int(outgoing_shortlist) <= 0 else min(int(outgoing_shortlist), len(contributions))
        outgoing_np = [member for _, member in contributions[:k_out]]

        union = list(dict.fromkeys(selected + incoming_np))
        union_pos = {value: idx for idx, value in enumerate(union)}
        union_gamma = gamma[union].to(dtype=torch.float64)
        gram = union_gamma @ union_gamma.t()
        target_dot = union_gamma @ v_target.to(dtype=torch.float64)

        best_gain = 0.0
        best_swap: Optional[Tuple[int, int, float]] = None
        for x_out in outgoing_np:
            out_pos = selected.index(int(x_out))
            for x_in in incoming_np:
                if int(x_in) in selected_set:
                    continue
                trial = list(selected)
                trial[out_pos] = int(x_in)
                subset = [union_pos[x] for x in trial]
                obj = _projector_objective_from_gram(gram=gram, target_dot=target_dot, subset=subset, eig_tol=eig_tol)
                gain = obj - current_obj
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (int(x_out), int(x_in), obj)

        best_gains_by_round.append(float(best_gain))
        if best_swap is None or best_gain <= float(improvement_tol):
            break
        x_out, x_in, new_obj = best_swap
        selected[selected.index(x_out)] = x_in
        current_obj = float(new_obj)
        accepted_gains.append(float(best_gain))

    q_rows = _mgs_basis_rows(gamma[selected], eps=max(eps, 1e-10))
    final_obj = _projector_objective_from_basis(q_rows, v_target)
    return np.asarray(selected, dtype=np.int64), {
        "initial_objective": float(initial_obj),
        "final_objective": float(final_obj),
        "accepted_swaps": int(len(accepted_gains)),
        "swap_rounds_run": int(len(best_gains_by_round)),
        "accepted_gains": accepted_gains,
        "best_gains_by_round": best_gains_by_round,
        "incoming_shortlist": int(incoming_shortlist),
        "outgoing_shortlist": int(outgoing_shortlist),
        "improvement_tol": float(improvement_tol),
    }


class OursCorrResidualRefineStrategy(BaseAcquisition):
    method_name = "ours_corr_residual_refine"

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
        attack_type, attack_steps, attack_step_size, attack_random_start = _secant_attack_settings(self.cfg)

        t_score = time.perf_counter()
        parts = compute_corr_residual_embeddings(
            model=model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            epsilon_acq=float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            attack_type=attack_type,
            attack_steps=attack_steps,
            attack_step_size=attack_step_size,
            attack_random_start=attack_random_start,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t_score

        g_clean = parts["g_clean"]
        gamma = parts["gamma"]
        clean_norm = parts["clean_norm"]
        gamma_norm = parts["gamma_norm"]
        if g_clean.ndim != 2 or gamma.ndim != 2 or g_clean.shape != gamma.shape:
            raise ValueError(f"g_clean and gamma must both be [N,D], got {tuple(g_clean.shape)} and {tuple(gamma.shape)}")
        if clean_norm.ndim != 1 or clean_norm.numel() != g_clean.size(0):
            raise ValueError("clean_norm must be [N] and match g_clean rows")

        v_target, target_debug = _build_correction_target(
            gamma=gamma,
            target=str(getattr(self.cfg, "corr_residual_target", "mean")),
            eps=float(getattr(self.cfg, "tiny", 1e-8)),
        )
        clean_gate_percentile = float(getattr(self.cfg, "clean_gate_percentile", 0.0))
        candidate_local, gate_debug = clean_gradient_percentile_gate(
            clean_norms=clean_norm,
            drop_percentile=clean_gate_percentile,
            budget=budget,
        )
        gamma_candidates = gamma[torch.as_tensor(candidate_local, dtype=torch.long)].contiguous()
        clean_norm_candidates = clean_norm[torch.as_tensor(candidate_local, dtype=torch.long)].contiguous()

        score_chunk_size = int(getattr(self.cfg, "corr_residual_score_chunk_size", getattr(self.cfg, "logdet_adv_disp_score_chunk_size", 8192)))
        max_refine_rounds = int(getattr(self.cfg, "corr_residual_refine_max_rounds", 3))
        incoming_shortlist = int(getattr(self.cfg, "corr_residual_refine_incoming_shortlist", 64))
        outgoing_shortlist = int(getattr(self.cfg, "corr_residual_refine_outgoing_shortlist", 8))
        improvement_tol = float(getattr(self.cfg, "corr_residual_refine_improvement_tol", 1e-8))
        eps = float(getattr(self.cfg, "tiny", 1e-8))

        t_select = time.perf_counter()
        picked_forward, q_forward, forward_debug = residual_forward_select(
            gamma=gamma_candidates,
            clean_norms=clean_norm_candidates,
            v_target=v_target,
            budget=budget,
            score_chunk_size=score_chunk_size,
            eps=eps,
        )
        forward_time = time.perf_counter() - t_select

        t_refine = time.perf_counter()
        picked_refined, refine_debug = refine_by_residual_swaps(
            gamma=gamma_candidates,
            clean_norms=clean_norm_candidates,
            selected_local=picked_forward,
            v_target=v_target,
            score_chunk_size=score_chunk_size,
            max_rounds=max_refine_rounds,
            incoming_shortlist=incoming_shortlist,
            outgoing_shortlist=outgoing_shortlist,
            improvement_tol=improvement_tol,
            eps=eps,
        )
        refine_time = time.perf_counter() - t_refine
        selection_time = time.perf_counter() - t_select

        picked_original_local = candidate_local[picked_refined]
        selected = np.asarray(unlabeled_indices, dtype=np.int64)[picked_original_local]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("ours_corr_residual_refine selected duplicate unlabeled indices")
        if len(selected) != min(int(budget), int(len(unlabeled_indices))):
            raise ValueError(
                "ours_corr_residual_refine returned unexpected number of samples: "
                f"{len(selected)} vs {min(int(budget), int(len(unlabeled_indices)))}"
            )

        zero_gamma_fallback = bool(forward_debug.get("fallback_reason") == "zero_correction_target_or_atoms")
        selected_clean_mean = (
            float(clean_norm[torch.as_tensor(picked_original_local, dtype=torch.long)].mean().item())
            if len(picked_original_local) > 0
            else float("nan")
        )
        selected_gamma_mean = (
            float(gamma_norm[torch.as_tensor(picked_original_local, dtype=torch.long)].mean().item())
            if len(picked_original_local) > 0
            else float("nan")
        )
        final_rank = int(_mgs_basis_rows(gamma_candidates[picked_refined], eps=max(eps, 1e-10)).size(0))

        if progress_logger is not None:
            if gate_debug["enabled"]:
                progress_logger.log(
                    (
                        "[OURS_CORR_RESIDUAL_REFINE_GATE] "
                        f"p={gate_debug['drop_percentile']:.3f} "
                        f"pool={gate_debug['pool_size_before']}->{gate_debug['pool_size_after']} "
                        f"tau={gate_debug['tau']:.6f}"
                    ),
                    device=str(device),
                )
            progress_logger.log(
                (
                    "[OURS_CORR_RESIDUAL_REFINE] "
                    f"selected={int(len(selected))} rank={final_rank} "
                    f"J_forward={float(forward_debug['forward_objectives'][-1]) if forward_debug['forward_objectives'] else 0.0:.6f} "
                    f"J_refined={float(refine_debug['final_objective']):.6f} "
                    f"swaps={int(refine_debug['accepted_swaps'])} "
                    f"zero_gamma_fallback={zero_gamma_fallback}"
                ),
                device=str(device),
            )

        extras: Dict[str, Any] = {
            "method": self.method_name,
            "objective": "correction_residual_target_projection",
            "selector_mode": "residual_forward_pursuit_plus_swap_refinement",
            "gradient_embedding": "last_layer_weight_gradient_no_bias",
            "target_definition": target_debug["target_type"],
            "target_raw_norm": float(target_debug["target_raw_norm"]),
            "target_normalized": bool(target_debug["target_normalized"]),
            "pool_size": int(len(unlabeled_indices)),
            "candidate_pool_size": int(len(candidate_local)),
            "selected_size": int(len(selected)),
            "base_gradient_dim": int(g_clean.size(1)),
            "g_clean_shape": [int(g_clean.size(0)), int(g_clean.size(1))],
            "gamma_shape": [int(gamma.size(0)), int(gamma.size(1))],
            "v_R_shape": [int(v_target.numel())],
            "Q_B_shape": [int(final_rank), int(gamma.size(1))],
            "clean_gate_percentile": float(clean_gate_percentile),
            "clean_gate_enabled": bool(gate_debug["enabled"]),
            "clean_gate_tau": float(gate_debug["tau"]),
            "clean_gate_pool_size_before": int(gate_debug["pool_size_before"]),
            "clean_gate_pool_size_after": int(gate_debug["pool_size_after"]),
            "clean_gate_requested_drop_count": int(gate_debug["requested_drop_count"]),
            "clean_gate_actual_drop_count": int(gate_debug["actual_drop_count"]),
            "clean_gate_score_stats": gate_debug["score_stats"],
            "clean_gate_retained_score_stats": gate_debug["retained_score_stats"],
            "attack_type": attack_type,
            "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
            "attack_steps": int(attack_steps),
            "attack_step_size": attack_step_size,
            "attack_random_start": bool(attack_random_start),
            "g_clean_norm_stats": tensor_stats(clean_norm),
            "g_adv_norm_stats": tensor_stats(parts["adv_norm"]),
            "gamma_norm_stats": tensor_stats(gamma_norm),
            "selected_mean_clean_grad_norm": selected_clean_mean,
            "selected_mean_gamma_norm": selected_gamma_mean,
            "zero_gamma_fallback": zero_gamma_fallback,
            "zero_gamma_fallback_behavior": "clean-gated top clean-gradient-norm selection",
            "score_chunk_size": int(score_chunk_size),
            "forward_objectives": forward_debug.get("forward_objectives", []),
            "forward_residual_norms": forward_debug.get("residual_norms", []),
            "forward_selected_scores": forward_debug.get("selected_forward_scores", []),
            "forward_basis_rank": int(forward_debug.get("basis_rank", 0)),
            "forward_fallback_count": int(forward_debug.get("fallback_count", 0)),
            "forward_time_sec": float(forward_time),
            "refinement_mode": "1_for_1_swap_shortlist",
            "refinement_initial_objective": float(refine_debug["initial_objective"]),
            "refined_final_objective": float(refine_debug["final_objective"]),
            "refinement_improvement": float(refine_debug["final_objective"] - refine_debug["initial_objective"]),
            "refinement_accepted_swaps": int(refine_debug["accepted_swaps"]),
            "refinement_rounds_run": int(refine_debug["swap_rounds_run"]),
            "refinement_accepted_gains": refine_debug["accepted_gains"],
            "refinement_best_gains_by_round": refine_debug["best_gains_by_round"],
            "refinement_incoming_shortlist": int(refine_debug["incoming_shortlist"]),
            "refinement_outgoing_shortlist": int(refine_debug["outgoing_shortlist"]),
            "refinement_improvement_tol": float(refine_debug["improvement_tol"]),
            "refinement_time_sec": float(refine_time),
            "selected_indices": selected.astype(np.int64).tolist(),
            "selected_local_indices": picked_original_local.astype(np.int64).tolist(),
            "selected_candidate_local_indices": picked_refined.astype(np.int64).tolist(),
        }
        return AcquisitionOutput(
            selected_indices=selected.astype(np.int64),
            scores=None,
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras=extras,
        )

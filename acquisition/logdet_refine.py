import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.t())


def _inverse_spd_with_jitter(
    matrix: torch.Tensor,
    jitter: float,
    max_tries: int = 6,
) -> Tuple[torch.Tensor, float, bool]:
    matrix = _symmetrize(matrix)
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    used = 0.0
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + used * eye)
            return torch.cholesky_inverse(chol), float(used), False
        except RuntimeError:
            used = float(jitter) if used == 0.0 else float(used * 10.0)
    fallback = max(float(jitter), float(used))
    return torch.linalg.pinv(matrix + fallback * eye), float(fallback), True


def _logdet_spd(
    matrix: torch.Tensor,
    jitter: float,
    max_tries: int = 6,
) -> Tuple[float, float, bool]:
    matrix = _symmetrize(matrix)
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    used = 0.0
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + used * eye)
            return float((2.0 * torch.log(torch.diag(chol))).sum().item()), float(used), False
        except RuntimeError:
            used = float(jitter) if used == 0.0 else float(used * 10.0)
    fallback = max(float(jitter), float(used))
    sign, logabsdet = torch.linalg.slogdet(matrix + fallback * eye)
    if float(sign.item()) <= 0.0:
        return float("-inf"), float(fallback), True
    return float(logabsdet.item()), float(fallback), True


def build_gram_matrix(
    embeddings: torch.Tensor,
    chunk_size: int = 1024,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build K = Phi Phi^T. This keeps the log-det selector in the dual, which is
    essential for last-layer gradient embeddings whose ambient dimension can be
    much larger than the acquisition batch size.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N,D], got shape={tuple(embeddings.shape)}")
    target_device = torch.device("cpu") if device is None else device
    phi = embeddings.to(device=target_device, dtype=dtype)
    n = int(phi.size(0))
    if n == 0:
        return torch.empty((0, 0), dtype=dtype, device=target_device)

    block = n if chunk_size <= 0 else max(1, int(chunk_size))
    kernel = torch.empty((n, n), dtype=dtype, device=target_device)
    phi_t = phi.t()
    for start in range(0, n, block):
        end = min(start + block, n)
        kernel[start:end] = phi[start:end] @ phi_t
    return _symmetrize(kernel)


def logdet_objective_from_gram(
    kernel: torch.Tensor,
    selected: np.ndarray,
    lambda_reg: float,
    ambient_dim: int = 0,
    jitter: float = 1e-8,
) -> float:
    if float(lambda_reg) <= 0.0:
        raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
    selected_t = torch.as_tensor(selected, dtype=torch.long, device=kernel.device)
    const = float(int(ambient_dim) * math.log(float(lambda_reg)))
    if selected_t.numel() == 0:
        return const
    k_sel = kernel[selected_t][:, selected_t]
    dual = torch.eye(selected_t.numel(), dtype=kernel.dtype, device=kernel.device)
    dual = dual + k_sel / float(lambda_reg)
    dual_obj, _, _ = _logdet_spd(dual, jitter=float(jitter))
    return const + float(dual_obj)


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
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        schur = 1.0 + diag[start:end] / float(lambda_reg)
        if selected.numel() > 0:
            k_chunk = kernel[start:end, selected] / float(lambda_reg)
            quad = (k_chunk @ a_inv * k_chunk).sum(dim=1)
            schur = schur - quad
        scores[start:end] = schur - 1.0
    return scores


def _rebuild_dual_state(
    kernel: torch.Tensor,
    selected: torch.Tensor,
    lambda_reg: float,
    jitter: float,
    ambient_dim: int,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    if selected.numel() == 0:
        empty_inv = torch.empty((0, 0), dtype=kernel.dtype, device=kernel.device)
        return empty_inv, float(int(ambient_dim) * math.log(float(lambda_reg))), {
            "inv_jitter": 0.0,
            "inv_used_pinv": False,
            "logdet_jitter": 0.0,
            "logdet_fallback": False,
        }
    dual = torch.eye(selected.numel(), dtype=kernel.dtype, device=kernel.device)
    dual = dual + kernel[selected][:, selected] / float(lambda_reg)
    a_inv, inv_jitter, inv_used_pinv = _inverse_spd_with_jitter(dual, jitter=float(jitter))
    dual_obj, logdet_jitter, logdet_fallback = _logdet_spd(dual, jitter=float(jitter))
    const = float(int(ambient_dim) * math.log(float(lambda_reg)))
    return a_inv, const + float(dual_obj), {
        "inv_jitter": float(inv_jitter),
        "inv_used_pinv": bool(inv_used_pinv),
        "logdet_jitter": float(logdet_jitter),
        "logdet_fallback": bool(logdet_fallback),
    }


def forward_greedy_select(
    kernel: torch.Tensor,
    query_size: int,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    ambient_dim: int = 0,
    progress_logger=None,
    progress_method_name: str = "LOGDET_FORWARD",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Forward greedy maximization of log det(lambda I + sum phi phi^T), evaluated
    in the dual via K = Phi Phi^T.
    """
    if kernel.ndim != 2 or kernel.size(0) != kernel.size(1):
        raise ValueError(f"kernel must be square [N,N], got shape={tuple(kernel.shape)}")
    if float(lambda_reg) <= 0.0:
        raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
    if float(jitter) <= 0.0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    k_mat = _symmetrize(kernel.to(dtype=torch.float64))
    n = int(k_mat.size(0))
    budget = min(int(query_size), n)
    const = float(int(ambient_dim) * math.log(float(lambda_reg)))
    if budget <= 0 or n == 0:
        return np.array([], dtype=np.int64), {
            "selected_scores": [],
            "selected_log_marginal_gains": [],
            "forward_objectives": [const],
            "initial_logdet_objective": const,
            "final_logdet_objective": const,
            "dual_final_logdet_objective": 0.0,
            "nonfinite_score_steps": 0,
            "inverse_rebuilds": 0,
            "used_pinv_rebuilds": 0,
            "max_jitter_used": 0.0,
        }

    diag = torch.diag(k_mat).clamp_min(0.0)
    selected_mask = torch.zeros((n,), dtype=torch.bool, device=k_mat.device)
    selected = torch.empty((0,), dtype=torch.long, device=k_mat.device)
    selected_list = []
    selected_scores = []
    selected_log_marginal_gains = []
    forward_objectives = [const]
    nonfinite_score_steps = 0
    inverse_rebuilds = 0
    used_pinv_rebuilds = 0
    max_jitter_used = 0.0
    a_inv: Optional[torch.Tensor] = None
    dual_obj = 0.0
    t0 = time.perf_counter()

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
            best_score = 0.0

        schur = max(1.0 + best_score, float(jitter))
        log_gain = float(math.log(schur))
        old_selected = selected
        selected_mask[best_local] = True
        selected_list.append(best_local)
        selected = torch.as_tensor(selected_list, dtype=torch.long, device=k_mat.device)
        selected_scores.append(float(best_score))
        selected_log_marginal_gains.append(log_gain)
        dual_obj += log_gain
        forward_objectives.append(const + float(dual_obj))

        if a_inv is None or old_selected.numel() == 0:
            a_inv = torch.empty((1, 1), dtype=k_mat.dtype, device=k_mat.device)
            a_inv[0, 0] = 1.0 / schur
        elif not math.isfinite(schur) or schur <= float(jitter):
            dual = torch.eye(selected.numel(), dtype=k_mat.dtype, device=k_mat.device)
            dual = dual + k_mat[selected][:, selected] / float(lambda_reg)
            a_inv, used_jitter, used_pinv = _inverse_spd_with_jitter(dual, jitter=float(jitter))
            inverse_rebuilds += 1
            max_jitter_used = max(max_jitter_used, float(used_jitter))
            used_pinv_rebuilds += int(used_pinv)
        else:
            b_vec = k_mat[old_selected, best_local] / float(lambda_reg)
            v = a_inv @ b_vec
            old_m = int(a_inv.size(0))
            new_inv = torch.empty((old_m + 1, old_m + 1), dtype=k_mat.dtype, device=k_mat.device)
            new_inv[:old_m, :old_m] = a_inv + torch.outer(v, v) / schur
            new_inv[:old_m, old_m] = -v / schur
            new_inv[old_m, :old_m] = -v / schur
            new_inv[old_m, old_m] = 1.0 / schur
            a_inv = _symmetrize(new_inv)

        if progress_logger is not None and ((step + 1) % 10 == 0 or (step + 1) == budget):
            progress_logger.log(
                (
                    f"[{progress_method_name}] step={step + 1}/{budget} "
                    f"gain={log_gain:.6e} objective={forward_objectives[-1]:.6f} "
                    f"elapsed={time.perf_counter() - t0:.2f}s"
                ),
                device=str(k_mat.device),
            )

    selected_np = np.asarray(selected_list, dtype=np.int64)
    final_obj = logdet_objective_from_gram(
        kernel=k_mat,
        selected=selected_np,
        lambda_reg=float(lambda_reg),
        ambient_dim=int(ambient_dim),
        jitter=float(jitter),
    )
    return selected_np, {
        "selected_scores": [float(x) for x in selected_scores],
        "selected_log_marginal_gains": [float(x) for x in selected_log_marginal_gains],
        "forward_objectives": [float(x) for x in forward_objectives],
        "initial_logdet_objective": float(const),
        "final_logdet_objective": float(final_obj),
        "dual_final_logdet_objective": float(final_obj - const),
        "nonfinite_score_steps": int(nonfinite_score_steps),
        "inverse_rebuilds": int(inverse_rebuilds),
        "used_pinv_rebuilds": int(used_pinv_rebuilds),
        "max_jitter_used": float(max_jitter_used),
    }


def refine_by_swap(
    kernel: torch.Tensor,
    selected_local: np.ndarray,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    ambient_dim: int = 0,
    max_swap_rounds: int = 3,
    swap_top_unselected: int = 200,
    swap_top_selected: int = 0,
    swap_improvement_tol: float = 1e-8,
    progress_logger=None,
    progress_method_name: str = "LOGDET_SWAP_REFINE",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministic 1-for-1 local refinement. A swap is accepted only after direct
    objective recomputation confirms improvement, so the returned set is never
    worse than the input set under the same log-det objective.
    """
    k_mat = _symmetrize(kernel.to(dtype=torch.float64))
    n = int(k_mat.size(0))
    selected = torch.as_tensor(selected_local, dtype=torch.long, device=k_mat.device)
    selected = torch.unique(selected, sorted=False)
    if n == 0 or selected.numel() == 0:
        return selected.detach().cpu().numpy().astype(np.int64), {
            "initial_logdet_objective": float("nan"),
            "final_logdet_objective": float("nan"),
            "accepted_swaps": 0,
            "swap_rounds_run": 0,
            "best_swap_gains_by_round": [],
            "accepted_swap_gains": [],
            "refinement_non_decrease": True,
        }

    diag = torch.diag(k_mat).clamp_min(0.0)
    selected_mask = torch.zeros((n,), dtype=torch.bool, device=k_mat.device)
    selected_mask[selected] = True
    a_inv, current_obj, state_debug = _rebuild_dual_state(
        kernel=k_mat,
        selected=selected,
        lambda_reg=float(lambda_reg),
        jitter=float(jitter),
        ambient_dim=int(ambient_dim),
    )
    initial_obj = float(current_obj)
    max_jitter_used = max(float(state_debug["inv_jitter"]), float(state_debug["logdet_jitter"]))
    accepted_swaps = 0
    accepted_swap_gains = []
    best_swap_gains_by_round = []
    state_rebuilds = 1
    state_used_pinv_rebuilds = int(state_debug["inv_used_pinv"])
    nonfinite_swap_score_events = 0

    rounds = max(0, int(max_swap_rounds))
    for swap_round in range(rounds):
        unselected = torch.nonzero(~selected_mask, as_tuple=False).squeeze(1)
        if unselected.numel() == 0:
            break

        add_scores = _kernel_quadratic_scores(
            kernel=k_mat,
            diag=diag,
            selected=selected,
            a_inv=a_inv,
            lambda_reg=float(lambda_reg),
            score_chunk_size=int(score_chunk_size),
        )
        add_scores = torch.where(torch.isfinite(add_scores), add_scores, torch.full_like(add_scores, -torch.inf))
        add_scores = add_scores.masked_fill(selected_mask, -torch.inf)
        if int(swap_top_unselected) > 0 and unselected.numel() > int(swap_top_unselected):
            top_u = min(int(swap_top_unselected), int(unselected.numel()))
            top_u_idx = torch.topk(add_scores[unselected], k=top_u, largest=True).indices
            candidate_unselected = unselected[top_u_idx]
        else:
            candidate_unselected = unselected

        inv_diag = torch.diag(a_inv).clamp_min(float(jitter))
        removal_losses = -torch.log(inv_diag)
        if int(swap_top_selected) > 0 and selected.numel() > int(swap_top_selected):
            top_b = min(int(swap_top_selected), int(selected.numel()))
            candidate_pos = torch.topk(removal_losses, k=top_b, largest=False).indices
        else:
            candidate_pos = torch.arange(selected.numel(), dtype=torch.long, device=k_mat.device)

        best_gain = float("-inf")
        best_remove_pos = None
        best_add = None

        for remove_pos_t in candidate_pos.tolist():
            remove_pos = int(remove_pos_t)
            remove_local = int(selected[remove_pos].item())
            keep_pos = torch.arange(selected.numel(), dtype=torch.long, device=k_mat.device)
            keep_pos = keep_pos[keep_pos != remove_pos]
            selected_minus = selected[keep_pos]
            alpha = float(a_inv[remove_pos, remove_pos].item())
            if not math.isfinite(alpha) or alpha <= float(jitter):
                a_minus_inv, obj_minus, minus_debug = _rebuild_dual_state(
                    kernel=k_mat,
                    selected=selected_minus,
                    lambda_reg=float(lambda_reg),
                    jitter=float(jitter),
                    ambient_dim=int(ambient_dim),
                )
                max_jitter_used = max(max_jitter_used, float(minus_debug["inv_jitter"]), float(minus_debug["logdet_jitter"]))
                state_used_pinv_rebuilds += int(minus_debug["inv_used_pinv"])
            elif selected_minus.numel() == 0:
                a_minus_inv = torch.empty((0, 0), dtype=k_mat.dtype, device=k_mat.device)
                obj_minus = float(int(ambient_dim) * math.log(float(lambda_reg)))
            else:
                block = a_inv[keep_pos][:, keep_pos]
                col = a_inv[keep_pos, remove_pos]
                a_minus_inv = block - torch.outer(col, col) / alpha
                a_minus_inv = _symmetrize(a_minus_inv)
                obj_minus = float(current_obj + math.log(alpha))

            cand = candidate_unselected
            if selected_minus.numel() > 0:
                k_chunk = k_mat[cand][:, selected_minus] / float(lambda_reg)
                quad = (k_chunk @ a_minus_inv * k_chunk).sum(dim=1)
                schur = 1.0 + diag[cand] / float(lambda_reg) - quad
            else:
                schur = 1.0 + diag[cand] / float(lambda_reg)
            finite = torch.isfinite(schur)
            if not bool(torch.all(finite)):
                nonfinite_swap_score_events += 1
            gains = torch.full_like(schur, -torch.inf)
            valid = finite & (schur > float(jitter))
            if bool(valid.any()):
                gains[valid] = float(obj_minus - current_obj) + torch.log(schur[valid])
            local_best = int(torch.argmax(gains).item())
            local_gain = float(gains[local_best].item())
            if math.isfinite(local_gain) and local_gain > best_gain:
                best_gain = local_gain
                best_remove_pos = remove_pos
                best_add = int(cand[local_best].item())

        best_swap_gains_by_round.append(float(best_gain) if math.isfinite(best_gain) else float("-inf"))
        if best_remove_pos is None or best_add is None or best_gain <= float(swap_improvement_tol):
            if progress_logger is not None:
                progress_logger.log(
                    (
                        f"[{progress_method_name}] round={swap_round + 1}/{rounds} "
                        f"best_gain={best_gain:.6e} stop"
                    ),
                    device=str(k_mat.device),
                )
            break

        proposed = selected.clone()
        old_remove = int(proposed[best_remove_pos].item())
        proposed[best_remove_pos] = int(best_add)
        proposed_np = proposed.detach().cpu().numpy().astype(np.int64)
        proposed_obj = logdet_objective_from_gram(
            kernel=k_mat,
            selected=proposed_np,
            lambda_reg=float(lambda_reg),
            ambient_dim=int(ambient_dim),
            jitter=float(jitter),
        )
        verified_gain = float(proposed_obj - current_obj)
        if verified_gain <= float(swap_improvement_tol):
            break

        selected_mask[old_remove] = False
        selected_mask[int(best_add)] = True
        selected = proposed
        a_inv, current_obj, rebuild_debug = _rebuild_dual_state(
            kernel=k_mat,
            selected=selected,
            lambda_reg=float(lambda_reg),
            jitter=float(jitter),
            ambient_dim=int(ambient_dim),
        )
        state_rebuilds += 1
        state_used_pinv_rebuilds += int(rebuild_debug["inv_used_pinv"])
        max_jitter_used = max(max_jitter_used, float(rebuild_debug["inv_jitter"]), float(rebuild_debug["logdet_jitter"]))
        accepted_swaps += 1
        accepted_swap_gains.append(verified_gain)

        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[{progress_method_name}] round={swap_round + 1}/{rounds} accepted "
                    f"remove={old_remove} add={best_add} gain={verified_gain:.6e} "
                    f"objective={current_obj:.6f}"
                ),
                device=str(k_mat.device),
            )

    return selected.detach().cpu().numpy().astype(np.int64), {
        "initial_logdet_objective": float(initial_obj),
        "final_logdet_objective": float(current_obj),
        "accepted_swaps": int(accepted_swaps),
        "swap_rounds_run": int(len(best_swap_gains_by_round)),
        "best_swap_gains_by_round": [float(x) for x in best_swap_gains_by_round],
        "accepted_swap_gains": [float(x) for x in accepted_swap_gains],
        "state_rebuilds": int(state_rebuilds),
        "state_used_pinv_rebuilds": int(state_used_pinv_rebuilds),
        "max_jitter_used": float(max_jitter_used),
        "nonfinite_swap_score_events": int(nonfinite_swap_score_events),
        "swap_top_unselected": int(swap_top_unselected),
        "swap_top_selected": int(swap_top_selected),
        "swap_improvement_tol": float(swap_improvement_tol),
        "refinement_non_decrease": bool(current_obj + float(jitter) >= initial_obj),
    }


def refine_by_backward_refill(
    kernel: torch.Tensor,
    selected_local: np.ndarray,
    lambda_reg: float = 1e-3,
    score_chunk_size: int = 8192,
    jitter: float = 1e-8,
    ambient_dim: int = 0,
    max_rounds: int = 3,
    top_removals: int = 10,
    progress_logger=None,
    progress_method_name: str = "LOGDET_BACKWARD_REFILL",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Backward deletion + refill is the same objective move as a restricted swap:
    shortlist weak selected points, remove one, and greedily refill from outside.
    """
    return refine_by_swap(
        kernel=kernel,
        selected_local=selected_local,
        lambda_reg=lambda_reg,
        score_chunk_size=score_chunk_size,
        jitter=jitter,
        ambient_dim=ambient_dim,
        max_swap_rounds=max_rounds,
        swap_top_unselected=0,
        swap_top_selected=max(1, int(top_removals)),
        swap_improvement_tol=1e-8,
        progress_logger=progress_logger,
        progress_method_name=progress_method_name,
    )

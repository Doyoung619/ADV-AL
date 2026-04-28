import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition


def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12, dim: int = -1) -> torch.Tensor:
    p = torch.clamp(probs, min=eps)
    return -(p * torch.log(p)).sum(dim=dim)


@torch.no_grad()
def get_mc_predictive_probs(
    model,
    unlabeled_loader,
    mc_passes: int,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
    progress_logger=None,
) -> torch.Tensor:
    """
    Collect predictive probabilities with shared MC pass index across the whole pool.
    Returns tensor with shape [N, K, C].
    """
    if device is None:
        device = next(model.parameters()).device
    if mc_passes <= 0:
        raise ValueError(f"mc_passes must be positive, got {mc_passes}")

    prev_training = model.training
    model.train()  # MC-dropout sampling
    per_pass_probs = []
    total_batches = len(unlabeled_loader)
    total_steps = max(1, mc_passes * max(1, total_batches))
    t0 = time.perf_counter()

    try:
        for mc_idx in range(mc_passes):
            pass_probs = []
            for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                pass_probs.append(probs.detach())

                if progress_logger is not None:
                    done = mc_idx * total_batches + batch_idx
                    if done % 20 == 0 or done == total_steps:
                        progress_logger.log_scoring_eta(
                            method="BatchBALD-MC",
                            processed_batches=done,
                            total_batches=total_steps,
                            elapsed=time.perf_counter() - t0,
                            device=str(device),
                        )
            per_pass_probs.append(torch.cat(pass_probs, dim=0))
    finally:
        model.train(prev_training)

    probs_nkc = torch.stack(per_pass_probs, dim=1)  # [N, K, C]
    return torch.clamp(probs_nkc, min=eps, max=1.0)


def compute_conditional_entropies(
    probs_nkc: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    probs = probs_nkc.to(dtype=torch.float32, device=probs_nkc.device)
    per_draw_entropy = _entropy_from_probs(probs, eps=eps, dim=2)  # [N, K]
    return per_draw_entropy.mean(dim=1)  # [N]


def compute_bald_scores_from_mc_probs(
    probs_nkc: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    probs = probs_nkc.to(dtype=torch.float32, device=probs_nkc.device)
    predictive = probs.mean(dim=1)  # [N, C]
    predictive_entropy = _entropy_from_probs(predictive, eps=eps, dim=1)
    conditional_entropy = compute_conditional_entropies(probs, eps=eps)
    return predictive_entropy - conditional_entropy


def compute_joint_entropy_exact(
    joint_prob_cache: torch.Tensor,
    candidate_probs_kc: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Exact entropy of (Y_A, Y_x) by enumerating all label configurations of Y_A.
    joint_prob_cache shape: [M, K] with M = C^|A|
    candidate_probs_kc shape: [K, C]
    """
    joint = joint_prob_cache.unsqueeze(2) * candidate_probs_kc.unsqueeze(0)  # [M, K, C]
    p_joint = joint.mean(dim=1).reshape(-1)  # [M*C]
    p_joint = torch.clamp(p_joint, min=eps)
    return float((-(p_joint * torch.log(p_joint))).sum().item())


def compute_joint_entropy_exact_batch(
    joint_prob_cache: torch.Tensor,
    candidate_probs_rkc: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized exact entropy for multiple candidates.
    joint_prob_cache: [M, K]
    candidate_probs_rkc: [R, K, C]
    returns: [R]
    """
    # [M, R, K, C]
    joint = joint_prob_cache[:, None, :, None] * candidate_probs_rkc[None, :, :, :]
    p_joint = joint.mean(dim=2)  # [M, R, C]
    p_joint = torch.clamp(p_joint, min=eps)
    ent = -(p_joint * torch.log(p_joint)).sum(dim=(0, 2))  # [R]
    return ent


def compute_joint_entropy_sampled(
    sampled_log_prob_cache: torch.Tensor,
    candidate_probs_kc: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Approximate entropy of (Y_A, Y_x) from sampled Y_A ~ p(Y_A).
    sampled_log_prob_cache shape: [S, K] with log p(y_A^s | w_k).
    """
    k = candidate_probs_kc.size(0)
    log_k = math.log(float(k))
    log_q = sampled_log_prob_cache  # [S, K]
    log_px = torch.log(torch.clamp(candidate_probs_kc, min=eps))  # [K, C]

    log_den = torch.logsumexp(log_q, dim=1, keepdim=True) - log_k  # [S, 1] = log p(y_A^s)
    log_num = torch.logsumexp(log_q.unsqueeze(2) + log_px.unsqueeze(0), dim=1) - log_k  # [S, C]
    cond = torch.exp(log_num - log_den)  # [S, C] = p(y_x|y_A^s)
    h_x_given_a = _entropy_from_probs(cond, eps=eps, dim=1).mean()  # E[H(y_x|y_A)]

    h_a_est = -log_den.mean()  # constant across candidates at same greedy step
    return float((h_a_est + h_x_given_a).item())


def compute_joint_entropy_sampled_batch(
    sampled_log_prob_cache: torch.Tensor,
    candidate_probs_rkc: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized sampled entropy for multiple candidates.
    sampled_log_prob_cache: [S, K]
    candidate_probs_rkc: [R, K, C]
    returns: [R]
    """
    k = candidate_probs_rkc.size(1)
    log_k = math.log(float(k))
    log_q = sampled_log_prob_cache  # [S, K]
    log_px = torch.log(torch.clamp(candidate_probs_rkc, min=eps))  # [R, K, C]

    log_den = torch.logsumexp(log_q, dim=1, keepdim=True) - log_k  # [S, 1]
    # [S, R, K, C] -> reduce K => [S, R, C]
    log_num = torch.logsumexp(log_q[:, None, :, None] + log_px[None, :, :, :], dim=2) - log_k
    cond = torch.exp(log_num - log_den[:, None, :])  # [S, R, C]
    h_x_given_a = _entropy_from_probs(cond, eps=eps, dim=2).mean(dim=0)  # [R]

    h_a_est = -log_den.mean()  # scalar
    return h_a_est + h_x_given_a


def update_joint_probability_cache(
    joint_prob_cache: torch.Tensor,
    selected_probs_kc: torch.Tensor,
) -> torch.Tensor:
    """
    Exact cache update:
      p(y_A, y_new | w) = p(y_A | w) * p(y_new | w)
    """
    expanded = joint_prob_cache.unsqueeze(2) * selected_probs_kc.unsqueeze(0)  # [M, K, C]
    # Flatten label-configuration axes first, keep posterior-sample axis (K) as columns.
    return expanded.permute(0, 2, 1).reshape(-1, selected_probs_kc.size(0))


def _sample_log_cache_from_exact(
    joint_prob_cache: torch.Tensor,
    num_joint_entropy_samples: int,
    seed: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    device = joint_prob_cache.device
    p_assign = joint_prob_cache.mean(dim=1)
    p_assign = torch.clamp(p_assign, min=eps)
    p_assign = p_assign / p_assign.sum()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sampled_idx = torch.multinomial(
        p_assign,
        num_samples=int(num_joint_entropy_samples),
        replacement=True,
        generator=generator,
    )
    sampled = joint_prob_cache[sampled_idx]
    return torch.log(torch.clamp(sampled, min=eps))


def _sample_conditional_labels(
    sampled_log_prob_cache: torch.Tensor,
    selected_probs_kc: torch.Tensor,
    seed: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Draw y_new^s ~ p(y_new | y_A^s) for each sampled configuration s.
    """
    k = selected_probs_kc.size(0)
    log_k = math.log(float(k))
    log_q = sampled_log_prob_cache
    log_px = torch.log(torch.clamp(selected_probs_kc, min=eps))
    log_den = torch.logsumexp(log_q, dim=1, keepdim=True) - log_k  # [S,1]
    log_num = torch.logsumexp(log_q.unsqueeze(2) + log_px.unsqueeze(0), dim=1) - log_k  # [S,C]
    cond = torch.exp(log_num - log_den)  # [S,C]
    cond = torch.clamp(cond, min=eps)
    cond = cond / cond.sum(dim=1, keepdim=True)

    generator = torch.Generator(device=selected_probs_kc.device)
    generator.manual_seed(seed)
    return torch.multinomial(cond, num_samples=1, replacement=True, generator=generator).squeeze(1)  # [S]


def _update_sampled_log_cache(
    sampled_log_prob_cache: torch.Tensor,
    selected_probs_kc: torch.Tensor,
    sampled_labels: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    log_px = torch.log(torch.clamp(selected_probs_kc, min=eps))  # [K,C]
    add = log_px[:, sampled_labels].t()  # [S,K]
    return sampled_log_prob_cache + add


def batchbald_greedy_select(
    probs_nkc: torch.Tensor,
    batch_size: int,
    num_joint_entropy_samples: int = 10000,
    exact_joint_entropy_steps: int = 4,
    candidate_chunk_size: int = 512,
    eps: float = 1e-12,
    seed: int = 0,
    progress_logger=None,
) -> Tuple[np.ndarray, dict]:
    probs = probs_nkc.to(dtype=torch.float32, device=probs_nkc.device)
    n, k, c = probs.shape
    if n == 0 or batch_size <= 0:
        return np.array([], dtype=np.int64), {"greedy_scores": []}

    budget = min(int(batch_size), int(n))
    conditional_entropies = compute_conditional_entropies(probs, eps=eps)  # [N]

    available = np.ones(n, dtype=bool)
    selected = []
    greedy_scores = []
    conditional_sum_selected = 0.0

    # Exact cache starts from empty set A = {}.
    joint_prob_cache = torch.ones((1, k), dtype=probs.dtype, device=probs.device)
    sampled_log_prob_cache = None

    # Separate seeds for cache-sampling and label-sampling updates.
    cache_seed = int(seed) * 1_000_003 + 17
    label_seed = int(seed) * 1_000_003 + 43
    label_update_count = 0

    for step in range(budget):
        if sampled_log_prob_cache is None and len(selected) >= int(exact_joint_entropy_steps):
            sampled_log_prob_cache = _sample_log_cache_from_exact(
                joint_prob_cache=joint_prob_cache,
                num_joint_entropy_samples=int(num_joint_entropy_samples),
                seed=cache_seed + step,
                eps=eps,
            )
            joint_prob_cache = None

        best_idx = None
        best_score = -float("inf")

        remaining = np.where(available)[0]
        for start in range(0, int(remaining.size), int(candidate_chunk_size)):
            rem_chunk = remaining[start : start + int(candidate_chunk_size)]
            cand = probs[torch.as_tensor(rem_chunk, device=probs.device, dtype=torch.long)]  # [R,K,C]
            if sampled_log_prob_cache is None:
                joint_ent = compute_joint_entropy_exact_batch(
                    joint_prob_cache=joint_prob_cache,
                    candidate_probs_rkc=cand,
                    eps=eps,
                )  # [R]
            else:
                joint_ent = compute_joint_entropy_sampled_batch(
                    sampled_log_prob_cache=sampled_log_prob_cache,
                    candidate_probs_rkc=cand,
                    eps=eps,
                )  # [R]

            cond_chunk = conditional_sum_selected + conditional_entropies[
                torch.as_tensor(rem_chunk, device=conditional_entropies.device, dtype=torch.long)
            ]
            score_chunk = joint_ent - cond_chunk
            chunk_best_local = int(torch.argmax(score_chunk).item())
            chunk_best_score = float(score_chunk[chunk_best_local].item())
            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_idx = int(rem_chunk[chunk_best_local])

        if best_idx is None:
            break

        selected.append(best_idx)
        greedy_scores.append(best_score)
        available[best_idx] = False
        conditional_sum_selected += float(conditional_entropies[best_idx].item())

        p_sel = probs[best_idx]  # [K,C]
        if sampled_log_prob_cache is None:
            joint_prob_cache = update_joint_probability_cache(joint_prob_cache=joint_prob_cache, selected_probs_kc=p_sel)
        else:
            sampled_labels = _sample_conditional_labels(
                sampled_log_prob_cache=sampled_log_prob_cache,
                selected_probs_kc=p_sel,
                seed=label_seed + label_update_count,
                eps=eps,
            )
            label_update_count += 1
            sampled_log_prob_cache = _update_sampled_log_cache(
                sampled_log_prob_cache=sampled_log_prob_cache,
                selected_probs_kc=p_sel,
                sampled_labels=sampled_labels,
                eps=eps,
            )

        if progress_logger is not None and ((step + 1) % 10 == 0 or (step + 1) == budget):
            mode = "exact" if sampled_log_prob_cache is None else "sampled"
            progress_logger.log(
                f"[BatchBALD] step={step + 1}/{budget} score={best_score:.6f} mode={mode}",
                device=str(probs.device),
            )

    debug = {
        "greedy_scores": greedy_scores,
        "num_candidates": int(n),
        "num_classes": int(c),
        "num_mc_passes": int(k),
        "exact_joint_entropy_steps": int(exact_joint_entropy_steps),
        "num_joint_entropy_samples": int(num_joint_entropy_samples),
    }
    return np.asarray(selected, dtype=np.int64), debug


class BatchBALDStrategy(BaseAcquisition):
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
        mc_probs = get_mc_predictive_probs(
            model=model,
            unlabeled_loader=unlabeled_loader,
            mc_passes=self.cfg.batchbald_num_mc_samples,
            device=device,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t_score

        t_sel = time.perf_counter()
        picked_local, debug = batchbald_greedy_select(
            probs_nkc=mc_probs,
            batch_size=budget,
            num_joint_entropy_samples=self.cfg.batchbald_num_joint_entropy_samples,
            exact_joint_entropy_steps=self.cfg.batchbald_exact_joint_entropy_steps,
            candidate_chunk_size=512,
            eps=1e-12,
            seed=self.cfg.seed,
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

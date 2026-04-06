import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

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
) -> torch.Tensor:
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
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    return torch.cat(displacements, dim=0)


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
        "nonfinite_score_steps": int(nonfinite_score_steps),
        "inverse_rebuilds": int(inverse_rebuilds),
        "used_pinv_rebuilds": int(used_pinv_rebuilds),
        "max_jitter_used": float(max_jitter_used),
    }


class LogDetAdvDispStrategy(BaseAcquisition):
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
        displacements = compute_adv_displacement_embeddings(
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
            progress_method_name="LOGDET_ADV_DISP_ATTACK",
            tensor_batch_size=self.cfg.pool_batch_size,
        )
        scoring_time = time.perf_counter() - t0

        if displacements.numel() > 0:
            first_step_scores_all = displacements.pow(2).sum(dim=1) / float(self.cfg.logdet_adv_disp_lambda)
            disp_norm_sq_all = displacements.pow(2).sum(dim=1)
        else:
            first_step_scores_all = torch.empty((0,), dtype=torch.float32, device=device)
            disp_norm_sq_all = first_step_scores_all

        percentile = float(self.cfg.logdet_adv_disp_percentile)
        if displacements.numel() > 0 and percentile > 0.0:
            threshold = torch.quantile(first_step_scores_all, q=percentile)
            feasible_mask = first_step_scores_all >= threshold
        else:
            threshold = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
            feasible_mask = torch.ones_like(first_step_scores_all, dtype=torch.bool)

        feasible_local = torch.nonzero(feasible_mask, as_tuple=False).squeeze(1)
        if feasible_local.numel() == 0:
            feasible_local = torch.arange(displacements.size(0), device=displacements.device, dtype=torch.long)
            feasible_mask = torch.ones_like(first_step_scores_all, dtype=torch.bool)

        displacements_sel = displacements[feasible_local]

        t_sel = time.perf_counter()
        picked_local, select_debug = greedy_logdet_selector(
            displacements=displacements_sel,
            query_size=budget,
            lambda_reg=self.cfg.logdet_adv_disp_lambda,
            score_chunk_size=self.cfg.logdet_adv_disp_score_chunk_size,
            jitter=self.cfg.logdet_adv_disp_jitter,
            progress_logger=progress_logger,
            progress_method_name="LOGDET_ADV_DISP_GREEDY",
        )
        selection_time = time.perf_counter() - t_sel

        picked_local_t = torch.as_tensor(picked_local, device=feasible_local.device, dtype=torch.long)
        picked_local_global = feasible_local[picked_local_t].detach().cpu().numpy()
        selected = unlabeled_indices[picked_local_global]

        first_step_scores = first_step_scores_all
        disp_norm_sq = disp_norm_sq_all
        disp_norm_sq_stats = tensor_stats(disp_norm_sq)
        first_step_score_stats = tensor_stats(first_step_scores)

        k = len(picked_local_global)
        selected_scores = select_debug["selected_scores"]
        selected_mean_score = float(np.mean(selected_scores)) if k > 0 else float("nan")

        debug_data = None
        if bool(getattr(self.cfg, "debug_save_adv_scores", False)):
            selected_flag = np.zeros(len(unlabeled_indices), dtype=np.int64)
            selected_flag[picked_local_global] = 1
            debug_data = {
                "__file_tag": "logdet_adv_disp_scores",
                "__column_order": [
                    "index",
                    "disp_norm_sq",
                    "first_step_score",
                    "feasible_flag",
                    "selected_flag",
                ],
                "index": np.asarray(unlabeled_indices, dtype=np.int64),
                "disp_norm_sq": disp_norm_sq.detach().cpu().numpy(),
                "first_step_score": first_step_scores.detach().cpu().numpy(),
                "feasible_flag": feasible_mask.detach().cpu().numpy().astype(np.int64),
                "selected_flag": selected_flag,
            }

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[LOGDET_ADV_DISP] disp_norm_sq_stats "
                    f"min={disp_norm_sq_stats['min']:.6f} max={disp_norm_sq_stats['max']:.6f} "
                    f"mean={disp_norm_sq_stats['mean']:.6f} std={disp_norm_sq_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[LOGDET_ADV_DISP] greedy_stats "
                    f"selected_mean_score={selected_mean_score:.6f} "
                    f"percentile={percentile:.3f} feasible={int(feasible_local.numel())}/{int(len(unlabeled_indices))} "
                    f"rebuilds={select_debug['inverse_rebuilds']} "
                    f"nonfinite_steps={select_debug['nonfinite_score_steps']}"
                ),
                device=str(device),
            )

        return AcquisitionOutput(
            selected_indices=selected,
            scores=first_step_scores.detach().cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": "logdet_adv_disp",
                "embedding": "logits",
                "objective": "logdet_lambdaI_plus_sum_adv_displacements",
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
                "percentile_threshold": float(threshold.item()) if torch.isfinite(threshold) else float("-inf"),
                "feasible_size": int(feasible_local.numel()),
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": selected.astype(np.int64).tolist(),
                "disp_norm_sq_stats": disp_norm_sq_stats,
                "first_step_score_stats": first_step_score_stats,
                "selected_mean_greedy_score": selected_mean_score,
                "selected_scores": select_debug["selected_scores"],
                "selected_log_marginal_gains": select_debug["selected_log_marginal_gains"],
                "nonfinite_score_steps": int(select_debug["nonfinite_score_steps"]),
                "inverse_rebuilds": int(select_debug["inverse_rebuilds"]),
                "used_pinv_rebuilds": int(select_debug["used_pinv_rebuilds"]),
                "max_jitter_used": float(select_debug["max_jitter_used"]),
            },
            debug_data=debug_data,
        )

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.utils import AcquisitionOutput, BaseAcquisition, topk_unlabeled_indices
from attacks import (
    fgsm_grad_disp_attack,
    fgsm_logit_mismatch_attack,
    fgsm_predictive_ce_attack,
    grad_disp_score_from_logits_features,
    pgd_grad_disp_attack,
    pgd_gap_attack,
    pgd_logit_mismatch_attack,
    pgd_predictive_ce_attack,
)


def _build_adv_examples(
    model,
    images: torch.Tensor,
    epsilon: float,
    mean,
    std,
    attack: str,
    pgd_steps: int,
    pgd_alpha: float,
    delta_objective: str,
) -> torch.Tensor:
    if attack == "pgd":
        if delta_objective == "predictive_ce":
            return pgd_predictive_ce_attack(
                model,
                images,
                epsilon=epsilon,
                alpha=pgd_alpha,
                steps=pgd_steps,
                mean=mean,
                std=std,
                random_start=True,
            )
        return pgd_logit_mismatch_attack(
            model,
            images,
            epsilon=epsilon,
            alpha=pgd_alpha,
            steps=pgd_steps,
            mean=mean,
            std=std,
            random_start=True,
        )

    if delta_objective == "predictive_ce":
        return fgsm_predictive_ce_attack(model, images, epsilon=epsilon, mean=mean, std=std)
    return fgsm_logit_mismatch_attack(model, images, epsilon=epsilon, mean=mean, std=std)


def _curvature_adaptive_score(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    """
    Compute per-sample curvature-adaptive score:
      score = Delta z^T (H + lambda I) Delta z
      H = diag(p) - p p^T, p = softmax(clean_logits)

    Shapes:
      clean_logits, adv_logits: [B, C]
      return: [B]
    """
    delta = adv_logits - clean_logits  # [B, C]
    probs = F.softmax(clean_logits, dim=1)  # [B, C]

    # delta^T H delta = sum_c p_c * delta_c^2 - (sum_c p_c * delta_c)^2
    weighted_sq = (probs * delta.pow(2)).sum(dim=1)  # [B]
    weighted_mean = (probs * delta).sum(dim=1)  # [B]
    hessian_quad = weighted_sq - weighted_mean.pow(2)  # [B]

    # delta^T (lambda I) delta = lambda * ||delta||_2^2
    l2_quad = lambda_reg * delta.pow(2).sum(dim=1)  # [B]
    return hessian_quad + l2_quad


def _logit_gap(
    logits: torch.Tensor,
    fixed_top2_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if fixed_top2_idx is None:
        top2_vals = torch.topk(logits, k=2, dim=1).values
        return top2_vals[:, 0] - top2_vals[:, 1]

    z1 = logits.gather(1, fixed_top2_idx[:, 0:1]).squeeze(1)
    z2 = logits.gather(1, fixed_top2_idx[:, 1:2]).squeeze(1)
    return z1 - z2


def _gap_change_score(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    use_fixed_clean_classes: bool = True,
) -> torch.Tensor:
    """
    score(x) = |g(x_adv) - g(x)|^2
      g(v) = top1_logit(v) - top2_logit(v)

    Default uses clean top-1/top-2 classes fixed for both clean/adv gaps.
    This is more stable than re-selecting top classes at each PGD step because
    it avoids discontinuous class-switching noise in the objective.
    """
    clean_top2_idx = torch.topk(clean_logits, k=2, dim=1).indices
    fixed_idx = clean_top2_idx if use_fixed_clean_classes else None
    clean_gap = _logit_gap(clean_logits, fixed_top2_idx=fixed_idx)
    adv_gap = _logit_gap(adv_logits, fixed_top2_idx=fixed_idx)
    return (adv_gap - clean_gap).pow(2)


def score_ours(
    model,
    unlabeled_loader,
    device: Optional[torch.device],
    epsilon: float,
    mean,
    std,
    attack: str = "fgsm",
    pgd_steps: int = 3,
    pgd_alpha: float = 2.0 / 255.0,
    delta_objective: str = "logit_mismatch",
    progress_logger=None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_scores = []
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            clean_logits = model(images)

        # PGD acquisition scoring is the main cost for this method in paper mode.
        adv = _build_adv_examples(
            model=model,
            images=images,
            epsilon=epsilon,
            mean=mean,
            std=std,
            attack=attack,
            pgd_steps=pgd_steps,
            pgd_alpha=pgd_alpha,
            delta_objective=delta_objective,
        )

        with torch.no_grad():
            adv_logits = model(adv)
            # score(x) = ||z(x+delta)-z(x)||_2^2
            scores = (adv_logits - clean_logits).pow(2).sum(dim=1)
            all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="OURS",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


def score_ours_gap(
    model,
    unlabeled_loader,
    device: Optional[torch.device],
    epsilon: float,
    mean,
    std,
    pgd_steps: int = 3,
    pgd_alpha: float = 2.0 / 255.0,
    use_fixed_clean_classes: bool = True,
    progress_logger=None,
) -> torch.Tensor:
    """
    Gap-based acquisition:
      score(x) = max_{||delta||_inf<=eps} |g(x+delta)-g(x)|^2
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_scores = []
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            clean_logits = model(images)

        adv = pgd_gap_attack(
            model=model,
            images=images,
            epsilon=epsilon,
            alpha=pgd_alpha,
            steps=pgd_steps,
            mean=mean,
            std=std,
            use_fixed_clean_classes=use_fixed_clean_classes,
            random_start=True,
        )

        with torch.no_grad():
            adv_logits = model(adv)
            scores = _gap_change_score(
                clean_logits=clean_logits,
                adv_logits=adv_logits,
                use_fixed_clean_classes=use_fixed_clean_classes,
            )
            all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="OURS_GAP",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


def score_ours_grad_disp(
    model,
    unlabeled_loader,
    device: Optional[torch.device],
    epsilon: float,
    mean,
    std,
    attack: str = "pgd",
    pgd_steps: int = 3,
    pgd_alpha: float = 2.0 / 255.0,
    progress_logger=None,
) -> torch.Tensor:
    """
    Gradient-displacement acquisition:
      score(x) = max_{||delta||_inf<=eps} ||u_W(x+delta) - u_W(x)||_F^2
    where u_W is last-layer CE gradient with pseudo-label fixed from clean x.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_scores = []
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            clean_logits, clean_features = model(images, return_features=True)
            # Keep pseudo-label fixed from clean input for objective stability.
            pseudo_labels = clean_logits.argmax(dim=1)

        if attack == "pgd":
            adv = pgd_grad_disp_attack(
                model=model,
                images=images,
                epsilon=epsilon,
                alpha=pgd_alpha,
                steps=pgd_steps,
                mean=mean,
                std=std,
                random_start=True,
            )
        else:
            adv = fgsm_grad_disp_attack(
                model=model,
                images=images,
                epsilon=epsilon,
                mean=mean,
                std=std,
            )

        with torch.no_grad():
            adv_logits, adv_features = model(adv, return_features=True)
            scores = grad_disp_score_from_logits_features(
                clean_logits=clean_logits,
                clean_features=clean_features,
                adv_logits=adv_logits,
                adv_features=adv_features,
                pseudo_labels=pseudo_labels,
            )
            all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="OURS_GRAD_DISP",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


def score_ours_hessian(
    model,
    unlabeled_loader,
    device: Optional[torch.device],
    epsilon: float,
    mean,
    std,
    attack: str = "fgsm",
    pgd_steps: int = 3,
    pgd_alpha: float = 2.0 / 255.0,
    delta_objective: str = "logit_mismatch",
    hessian_lambda: float = 1e-3,
    progress_logger=None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_scores = []
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            clean_logits = model(images)

        adv = _build_adv_examples(
            model=model,
            images=images,
            epsilon=epsilon,
            mean=mean,
            std=std,
            attack=attack,
            pgd_steps=pgd_steps,
            pgd_alpha=pgd_alpha,
            delta_objective=delta_objective,
        )

        with torch.no_grad():
            adv_logits = model(adv)
            scores = _curvature_adaptive_score(
                clean_logits=clean_logits,
                adv_logits=adv_logits,
                lambda_reg=hessian_lambda,
            )
            all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method="OURS_HESSIAN",
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


class OursStrategy(BaseAcquisition):
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
        scores = score_ours(
            model,
            unlabeled_loader,
            device=device,
            epsilon=self.cfg.epsilon,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            attack=self.cfg.acquisition_attack,
            pgd_steps=self.cfg.acquisition_pgd_steps,
            pgd_alpha=self.cfg.acquisition_pgd_alpha,
            delta_objective=self.cfg.ours_delta_objective,
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
            extras={
                "attack": self.cfg.acquisition_attack,
                "delta_objective": self.cfg.ours_delta_objective,
                "epsilon": self.cfg.epsilon,
                "pgd_steps": self.cfg.acquisition_pgd_steps,
            },
        )


class OursHessianStrategy(BaseAcquisition):
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
        scores = score_ours_hessian(
            model,
            unlabeled_loader,
            device=device,
            epsilon=self.cfg.epsilon,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            attack=self.cfg.acquisition_attack,
            pgd_steps=self.cfg.acquisition_pgd_steps,
            pgd_alpha=self.cfg.acquisition_pgd_alpha,
            delta_objective=self.cfg.ours_delta_objective,
            hessian_lambda=self.cfg.ours_hessian_lambda,
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
            extras={
                "attack": self.cfg.acquisition_attack,
                "delta_objective": self.cfg.ours_delta_objective,
                "epsilon": self.cfg.epsilon,
                "pgd_steps": self.cfg.acquisition_pgd_steps,
                "ours_hessian_lambda": self.cfg.ours_hessian_lambda,
            },
        )


class OursGapStrategy(BaseAcquisition):
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
        scores = score_ours_gap(
            model,
            unlabeled_loader,
            device=device,
            epsilon=self.cfg.epsilon,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            pgd_steps=self.cfg.acquisition_pgd_steps,
            pgd_alpha=self.cfg.acquisition_pgd_alpha,
            use_fixed_clean_classes=self.cfg.ours_gap_use_fixed_clean_classes,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        k = min(budget, scores.numel())
        top_vals, top_local = torch.topk(scores, k=k, largest=True)
        selected = unlabeled_indices[top_local.cpu().numpy()]
        selection_time = time.perf_counter() - t1

        return AcquisitionOutput(
            selected_indices=selected,
            scores=scores.numpy(),
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras={
                "attack": "pgd",
                "objective": "gap_change_sq",
                "epsilon": self.cfg.epsilon,
                "pgd_steps": self.cfg.acquisition_pgd_steps,
                "pgd_alpha": self.cfg.acquisition_pgd_alpha,
                "ours_gap_use_fixed_clean_classes": self.cfg.ours_gap_use_fixed_clean_classes,
                "selected_mean_gap_score": float(top_vals.mean().item()) if k > 0 else float("nan"),
            },
        )


class OursGradDispStrategy(BaseAcquisition):
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
        scores = score_ours_grad_disp(
            model,
            unlabeled_loader,
            device=device,
            epsilon=self.cfg.epsilon,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            attack=self.cfg.acquisition_attack,
            pgd_steps=self.cfg.acquisition_pgd_steps,
            pgd_alpha=self.cfg.acquisition_pgd_alpha,
            progress_logger=progress_logger,
        )
        scoring_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        k = min(budget, scores.numel())
        top_vals, top_local = torch.topk(scores, k=k, largest=True)
        selected = unlabeled_indices[top_local.cpu().numpy()]
        selection_time = time.perf_counter() - t1

        return AcquisitionOutput(
            selected_indices=selected,
            scores=scores.numpy(),
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras={
                "attack": self.cfg.acquisition_attack,
                "objective": "grad_disp_sq",
                "epsilon": self.cfg.epsilon,
                "pgd_steps": self.cfg.acquisition_pgd_steps,
                "pgd_alpha": self.cfg.acquisition_pgd_alpha,
                "mean_grad_disp_score": float(scores.mean().item()) if scores.numel() > 0 else float("nan"),
                "selected_mean_grad_disp_score": float(top_vals.mean().item()) if k > 0 else float("nan"),
            },
        )

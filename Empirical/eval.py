from typing import Sequence

import torch
import torch.nn.functional as F

from attacks import (
    fgsm_attack,
    fgsm_logit_mismatch_attack,
    pgd_attack,
    pgd_logit_mismatch_attack,
)


@torch.no_grad()
def eval_avg_logit_norm_sq(model, loader, device: torch.device) -> float:
    """Average squared L2 norm of logits: E[||z(x)||_2^2]."""
    model.eval()
    total = 0
    total_norm_sq = 0.0
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
        norm_sq = logits.pow(2).sum(dim=1)
        total_norm_sq += norm_sq.sum().item()
        total += norm_sq.numel()
    return total_norm_sq / max(1, total)


def eval_round_core_metrics(
    model,
    loader,
    device: torch.device,
    epsilon: float,
    alpha: float,
    mean: Sequence[float],
    std: Sequence[float],
    pgd_steps: int = 20,
    curvature_lambda: float = 1e-3,
    gap_use_fixed_clean_classes: bool = True,
) -> dict:
    """
    Round-level metrics on test set:
    1) clean accuracy
    2) PGD-{pgd_steps} adversarial accuracy
    3) mean ||z(x)||_2
    4) mean margin = z_y - max_{j!=y} z_j
    5) mean absolute mismatch = mean ||z(x_adv)-z(x)||_2
    6) mean normalized mismatch = mean ||z(x_adv)-z(x)||_2 / (||z(x)||_2 + 1e-8)
    7) mean curvature score = mean Delta z^T (H + lambda I) Delta z
       where H = diag(p) - p p^T, p = softmax(z(x))
    8) mean gap score = mean |g(x_adv)-g(x)|^2, g(v)=top1(v)-top2(v)
    """
    model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0

    sum_logit_norm = 0.0
    sum_margin = 0.0
    sum_abs_mismatch = 0.0
    sum_norm_mismatch = 0.0
    sum_curvature_score = 0.0
    sum_gap_score = 0.0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()

            logit_norm = torch.norm(clean_logits, p=2, dim=1)  # [B]
            zy = clean_logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
            masked = clean_logits.clone()
            masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
            z_other = masked.max(dim=1).values
            margin = zy - z_other

        adv = pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=pgd_steps,
            mean=mean,
            std=std,
            random_start=True,
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                adv_logits = model(adv)
            adv_preds = adv_logits.argmax(dim=1)
            adv_correct += (adv_preds == labels).sum().item()

            delta_logits = adv_logits - clean_logits  # [B, C]
            mismatch = torch.norm(delta_logits, p=2, dim=1)  # [B]
            norm_mismatch = mismatch / (logit_norm + 1e-8)

            probs = F.softmax(clean_logits, dim=1)  # [B, C]
            weighted_sq = (probs * delta_logits.pow(2)).sum(dim=1)  # [B]
            weighted_mean = (probs * delta_logits).sum(dim=1)  # [B]
            hessian_quad = weighted_sq - weighted_mean.pow(2)  # [B]
            curvature_score = hessian_quad + float(curvature_lambda) * delta_logits.pow(2).sum(dim=1)

            clean_top2_idx = torch.topk(clean_logits, k=2, dim=1).indices
            if gap_use_fixed_clean_classes:
                clean_gap = clean_logits.gather(1, clean_top2_idx[:, 0:1]).squeeze(1) - clean_logits.gather(
                    1, clean_top2_idx[:, 1:2]
                ).squeeze(1)
                adv_gap = adv_logits.gather(1, clean_top2_idx[:, 0:1]).squeeze(1) - adv_logits.gather(
                    1, clean_top2_idx[:, 1:2]
                ).squeeze(1)
            else:
                clean_top2_vals = torch.topk(clean_logits, k=2, dim=1).values
                adv_top2_vals = torch.topk(adv_logits, k=2, dim=1).values
                clean_gap = clean_top2_vals[:, 0] - clean_top2_vals[:, 1]
                adv_gap = adv_top2_vals[:, 0] - adv_top2_vals[:, 1]
            gap_score = (adv_gap - clean_gap).pow(2)

        bsz = labels.size(0)
        total += bsz
        sum_logit_norm += logit_norm.sum().item()
        sum_margin += margin.sum().item()
        sum_abs_mismatch += mismatch.sum().item()
        sum_norm_mismatch += norm_mismatch.sum().item()
        sum_curvature_score += curvature_score.sum().item()
        sum_gap_score += gap_score.sum().item()

    return {
        "clean_acc": 100.0 * clean_correct / max(1, total),
        "pgd_robust_acc": 100.0 * adv_correct / max(1, total),
        "pgd20_acc": 100.0 * adv_correct / max(1, total),
        "mean_logit_norm": sum_logit_norm / max(1, total),
        "mean_margin": sum_margin / max(1, total),
        "mean_abs_mismatch": sum_abs_mismatch / max(1, total),
        "mean_normalized_mismatch": sum_norm_mismatch / max(1, total),
        "mean_curvature_score": sum_curvature_score / max(1, total),
        "mean_gap_score": sum_gap_score / max(1, total),
    }


@torch.no_grad()
def eval_clean_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


def eval_fgsm_accuracy(
    model,
    loader,
    device: torch.device,
    epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = fgsm_attack(model, images, labels, epsilon=epsilon, mean=mean, std=std)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(adv)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


def eval_pgd_accuracy(
    model,
    loader,
    device: torch.device,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            mean=mean,
            std=std,
            random_start=True,
        )
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(adv)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


def eval_avg_logit_mismatch(
    model,
    loader,
    device: torch.device,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    attack: str = "pgd",
) -> float:
    model.eval()
    total_mismatch = 0.0
    total = 0
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                clean_logits = model(images)

        if attack == "fgsm":
            adv = fgsm_logit_mismatch_attack(model, images, epsilon=epsilon, mean=mean, std=std)
        else:
            adv = pgd_logit_mismatch_attack(
                model,
                images,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                mean=mean,
                std=std,
                random_start=True,
            )

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                adv_logits = model(adv)
            mismatch = torch.norm(adv_logits - clean_logits, p=2, dim=1)
        total_mismatch += mismatch.sum().item()
        total += mismatch.numel()
    return total_mismatch / max(1, total)


def eval_avg_logit_mismatch_sq(
    model,
    loader,
    device: torch.device,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    attack: str = "pgd",
) -> float:
    """
    Average squared logit mismatch:
      E[ || z(x_adv) - z(x) ||_2^2 ]
    """
    model.eval()
    total_mismatch_sq = 0.0
    total = 0
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                clean_logits = model(images)

        if attack == "fgsm":
            adv = fgsm_logit_mismatch_attack(model, images, epsilon=epsilon, mean=mean, std=std)
        else:
            adv = pgd_logit_mismatch_attack(
                model,
                images,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                mean=mean,
                std=std,
                random_start=True,
            )

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                adv_logits = model(adv)
            mismatch_sq = (adv_logits - clean_logits).pow(2).sum(dim=1)
        total_mismatch_sq += mismatch_sq.sum().item()
        total += mismatch_sq.numel()
    return total_mismatch_sq / max(1, total)

import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from acquisition.badge import select_badge_kmeanspp
from acquisition.utils import AcquisitionOutput, BaseAcquisition, tensor_stats, topk_unlabeled_indices


def get_pseudo_label(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def build_perturbation_from_grad(
    grad_tensors: Sequence[Optional[torch.Tensor]],
    rho: float,
    norm: str = "linf",
    tiny: float = 1e-12,
) -> List[Optional[torch.Tensor]]:
    norm = norm.lower()
    if norm == "linf":
        return [None if g is None else float(rho) * g.sign() for g in grad_tensors]
    if norm == "l2":
        sq_sum = torch.zeros((), device=grad_tensors[0].device if grad_tensors else "cpu")
        for g in grad_tensors:
            if g is None:
                continue
            sq_sum = sq_sum + g.float().pow(2).sum()
        denom = sq_sum.sqrt().clamp_min(float(tiny))
        return [None if g is None else float(rho) * (g / denom) for g in grad_tensors]
    raise ValueError(f"Unsupported SAAL norm: {norm}")


@contextmanager
def temporarily_perturbed_model(
    params: Sequence[torch.nn.Parameter],
    perturbations: Sequence[Optional[torch.Tensor]],
):
    with torch.no_grad():
        for p, eps in zip(params, perturbations):
            if eps is not None:
                p.add_(eps)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, eps in zip(params, perturbations):
                if eps is not None:
                    p.sub_(eps)


def compute_single_sample_grad(
    model,
    x: torch.Tensor,
    pseudo_label: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, pseudo_label, reduction="mean")
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    return loss, list(grads)


def score_single_saal(
    model,
    x: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    rho: float,
    norm: str = "linf",
    return_debug: bool = False,
) -> Tuple[float, Optional[Dict[str, float]]]:
    if x.ndim == 3:
        x = x.unsqueeze(0)

    # Step A: pseudo-label from ORIGINAL model.
    logits_clean = model(x)
    pseudo_label = get_pseudo_label(logits_clean).detach()

    # Step B: per-sample gradient on unperturbed model.
    clean_loss, grads = compute_single_sample_grad(
        model=model,
        x=x,
        pseudo_label=pseudo_label,
        params=params,
    )

    # Step C: SAM-style parameter perturbation.
    perturbation = build_perturbation_from_grad(
        grad_tensors=grads,
        rho=rho,
        norm=norm,
    )

    # Step D: evaluate perturbed loss using fixed pseudo-label from step A.
    with temporarily_perturbed_model(params=params, perturbations=perturbation):
        logits_perturbed = model(x)
        perturbed_loss = F.cross_entropy(logits_perturbed, pseudo_label, reduction="mean")
        pseudo_from_perturbed = get_pseudo_label(logits_perturbed).detach()

    # Step E: model parameters are restored by context manager.
    score = float(perturbed_loss.detach().item())
    if not return_debug:
        return score, None
    debug = {
        "clean_loss": float(clean_loss.detach().item()),
        "perturbed_loss": float(perturbed_loss.detach().item()),
        "pseudo_label_used": int(pseudo_label.item()),
        "pseudo_label_original": int(pseudo_label.item()),
        "pseudo_label_perturbed": int(pseudo_from_perturbed.item()),
    }
    return score, debug


def _score_batchwise_saal(
    model,
    images: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    rho: float,
    norm: str,
) -> torch.Tensor:
    """
    Optional approximation:
    - build one perturbation from batch-mean CE with pseudo-labels
    - evaluate per-sample perturbed CE with fixed pseudo-labels
    """
    model.zero_grad(set_to_none=True)
    logits_clean = model(images)
    pseudo = get_pseudo_label(logits_clean).detach()
    clean_loss = F.cross_entropy(logits_clean, pseudo, reduction="mean")
    grads = torch.autograd.grad(
        clean_loss,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    perturbation = build_perturbation_from_grad(
        grad_tensors=grads,
        rho=rho,
        norm=norm,
    )
    with temporarily_perturbed_model(params=params, perturbations=perturbation):
        logits_perturbed = model(images)
        losses = F.cross_entropy(logits_perturbed, pseudo, reduction="none")
    return losses.detach()


@torch.no_grad()
def compute_feature_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    features = []
    for images, _, _ in unlabeled_loader:
        images = images.to(device, non_blocking=True)
        _, feats = model(images, return_features=True)
        features.append(feats.detach().cpu())
    return torch.cat(features, dim=0)


def score_saal(
    model,
    unlabeled_loader,
    rho: float = 0.05,
    norm: str = "linf",
    device: Optional[torch.device] = None,
    progress_logger=None,
    batchwise_perturb: bool = False,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    prev_training = model.training
    params = [p for p in model.parameters()]
    requires_grad_state = [bool(p.requires_grad) for p in params]
    for p in params:
        p.requires_grad_(True)
    model.eval()

    all_scores = []
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    try:
        for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
            images = images.to(device, non_blocking=True)

            if batchwise_perturb:
                batch_scores = _score_batchwise_saal(
                    model=model,
                    images=images,
                    params=params,
                    rho=rho,
                    norm=norm,
                )
                all_scores.append(batch_scores.cpu())
            else:
                per_sample_scores = []
                for i in range(images.size(0)):
                    x_i = images[i : i + 1]
                    s_i, _ = score_single_saal(
                        model=model,
                        x=x_i,
                        params=params,
                        rho=rho,
                        norm=norm,
                        return_debug=False,
                    )
                    per_sample_scores.append(s_i)
                all_scores.append(torch.tensor(per_sample_scores, dtype=torch.float32))

            if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
                progress_logger.log_scoring_eta(
                    method="SAAL",
                    processed_batches=batch_idx,
                    total_batches=total_batches,
                    elapsed=time.perf_counter() - t0,
                    device=str(device),
                )
    finally:
        model.train(prev_training)
        for p, req in zip(params, requires_grad_state):
            p.requires_grad_(req)
        model.zero_grad(set_to_none=True)

    return torch.cat(all_scores, dim=0)


class SAALStrategy(BaseAcquisition):
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

        t_score = time.perf_counter()
        scores = score_saal(
            model=model,
            unlabeled_loader=unlabeled_loader,
            rho=float(self.cfg.saal_rho),
            norm=self.cfg.saal_norm,
            device=device,
            progress_logger=progress_logger,
            batchwise_perturb=bool(getattr(self.cfg, "saal_batchwise_perturb", False)),
        ).float()
        scoring_time = time.perf_counter() - t_score

        t_select = time.perf_counter()
        selection_mode = "topk"
        if bool(self.cfg.saal_use_kmeanspp):
            candidate_target = max(int(budget), int(float(self.cfg.candidate_ratio) * float(budget)))
            candidate_k = min(int(scores.numel()), int(candidate_target))
            candidate_local = torch.topk(scores, k=candidate_k, largest=True).indices.cpu().numpy()

            features = compute_feature_embeddings(
                model=model,
                unlabeled_loader=unlabeled_loader,
                device=device,
            )
            candidate_features = features[candidate_local]
            picked_in_candidate = select_badge_kmeanspp(
                embeddings=candidate_features,
                B=int(budget),
                seed=int(self.cfg.seed),
            )
            picked_local = candidate_local[picked_in_candidate]
            selected = unlabeled_indices[picked_local]
            selection_mode = "topk_then_kmeanspp"
        else:
            selected = topk_unlabeled_indices(unlabeled_indices, scores, budget)
        selection_time = time.perf_counter() - t_select

        score_stats: Dict[str, float] = tensor_stats(scores)
        total_time = time.perf_counter() - total_t0

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[SAAL] stats "
                    f"min={score_stats['min']:.6f} max={score_stats['max']:.6f} "
                    f"mean={score_stats['mean']:.6f} std={score_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[SAAL] config "
                    f"rho={float(self.cfg.saal_rho):.6f} norm={self.cfg.saal_norm} "
                    f"use_kmeanspp={bool(self.cfg.saal_use_kmeanspp)} "
                    f"batchwise_perturb={bool(getattr(self.cfg, 'saal_batchwise_perturb', False))}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[SAAL] timing "
                    f"scoring={scoring_time:.3f}s selection={selection_time:.3f}s "
                    f"total={total_time:.3f}s pool={int(len(unlabeled_indices))} picked={int(len(selected))}"
                ),
                device=str(device),
            )

        return AcquisitionOutput(
            selected_indices=np.asarray(selected, dtype=np.int64),
            scores=scores.cpu().numpy(),
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": "saal",
                "rho": float(self.cfg.saal_rho),
                "norm": self.cfg.saal_norm,
                "selection_mode": selection_mode,
                "use_kmeanspp": bool(self.cfg.saal_use_kmeanspp),
                "batchwise_perturb": bool(getattr(self.cfg, "saal_batchwise_perturb", False)),
                "score_stats": score_stats,
                "pool_size": int(len(unlabeled_indices)),
                "selected_size": int(len(selected)),
                "selected_indices": np.asarray(selected, dtype=np.int64).tolist(),
                "timing_saal_scores_sec": float(scoring_time),
                "timing_selection_sec": float(selection_time),
                "timing_total_acquisition_sec": float(total_time),
            },
        )

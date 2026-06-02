import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from acquisition.badge import compute_badge_embeddings, select_badge_kmeanspp
from acquisition.utils import AcquisitionOutput, BaseAcquisition


def _channel_tensor(values: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def _scaled_linf_eps(
    epsilon: float,
    std: Optional[Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> torch.Tensor:
    if std is None:
        return torch.full((1, channels, 1, 1), float(epsilon), device=device, dtype=dtype)
    std_t = _channel_tensor(std, device=device, dtype=dtype)
    return torch.full((1, channels, 1, 1), float(epsilon), device=device, dtype=dtype) / std_t


def _clamp_to_valid_range(
    x: torch.Tensor,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> torch.Tensor:
    if mean is None or std is None:
        return x.clamp(0.0, 1.0)
    mean_t = _channel_tensor(mean, x.device, x.dtype)
    std_t = _channel_tensor(std, x.device, x.dtype)
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t
    return torch.max(torch.min(x, upper), lower)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def normalize_adv_scores(scores: torch.Tensor, mode: str, tiny: float) -> torch.Tensor:
    mode = mode.lower()
    x = scores.float().clamp_min(0.0)
    if mode == "none":
        return x
    if mode == "mean":
        return x / (x.mean() + tiny)
    if mode == "zscore_positive":
        z = (x - x.mean()) / (x.std(unbiased=False) + tiny)
        return torch.relu(z)
    if mode == "log_mean":
        lx = torch.log1p(x)
        return lx / (lx.mean() + tiny)
    raise ValueError(f"Unsupported adv_score_normalization: {mode}")


def compute_adv_logit_mismatch_scores(
    model,
    unlabeled_loader,
    epsilon_acq: float,
    attack_type: str = "fgsm",
    pgd_steps: int = 3,
    pgd_step_size: Optional[float] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    progress_logger=None,
    progress_method_name: str = "ADV_LOGIT_MISMATCH",
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    attack_type = attack_type.lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    model.eval()
    total_batches = len(unlabeled_loader)
    t0 = time.perf_counter()
    all_scores = []

    for batch_idx, (images, _, _) in enumerate(unlabeled_loader, start=1):
        images = images.to(device, non_blocking=True)
        x0 = images.detach()
        channels = x0.size(1)

        with torch.no_grad():
            z_clean = model(x0).detach()

        eps_t = _scaled_linf_eps(epsilon_acq, std, x0.device, x0.dtype, channels=channels)

        if attack_type == "pgd":
            steps = max(1, int(pgd_steps))
            step_size = float(pgd_step_size) if pgd_step_size is not None else float(epsilon_acq) / float(steps)
            alpha_t = _scaled_linf_eps(step_size, std, x0.device, x0.dtype, channels=channels)

            delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
            x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

            for _ in range(steps):
                x_adv = x_adv.detach().requires_grad_(True)
                z_adv = model(x_adv)
                mismatch_obj = (z_adv - z_clean).pow(2).sum(dim=1).mean()
                grad = torch.autograd.grad(mismatch_obj, x_adv, only_inputs=True)[0]

                x_adv = x_adv.detach() + alpha_t * grad.sign()
                delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
                x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

            x_adv = x_adv.detach()
        else:
            x_adv = x0.clone().detach().requires_grad_(True)
            z_adv = model(x_adv)
            mismatch_obj = (z_adv - z_clean).pow(2).sum(dim=1).mean()
            grad = torch.autograd.grad(mismatch_obj, x_adv, only_inputs=True)[0]
            x_adv = _clamp_to_valid_range(x0 + eps_t * grad.sign(), mean, std).detach()

        with torch.no_grad():
            z_adv_final = model(x_adv)
            scores = (z_adv_final - z_clean).pow(2).sum(dim=1)
            all_scores.append(scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=progress_method_name,
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)


class BADGEAdvMultStrategy(BaseAcquisition):
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

        t_badge = time.perf_counter()
        badge_embeddings = compute_badge_embeddings(
            model,
            unlabeled_loader,
            device=device,
            projection_dim=self.cfg.badge_projection_dim,
            seed=self.cfg.seed,
            progress_logger=progress_logger,
        )
        badge_time = time.perf_counter() - t_badge

        t_adv = time.perf_counter()
        raw_adv_scores = compute_adv_logit_mismatch_scores(
            model=model,
            unlabeled_loader=unlabeled_loader,
            epsilon_acq=self.cfg.epsilon_acq,
            attack_type=self.cfg.adv_attack_type_for_acquisition,
            pgd_steps=self.cfg.adv_pgd_steps,
            pgd_step_size=self.cfg.adv_pgd_step_size,
            mean=self.cfg.cifar10_mean,
            std=self.cfg.cifar10_std,
            device=device,
            progress_logger=progress_logger,
            progress_method_name="BADGE_ADV_MULT_ADV",
        )
        adv_time = time.perf_counter() - t_adv

        t_norm = time.perf_counter()
        norm_adv_scores = normalize_adv_scores(
            raw_adv_scores,
            mode=self.cfg.adv_score_normalization,
            tiny=self.cfg.adv_tiny,
        )
        weights = torch.sqrt(norm_adv_scores + self.cfg.adv_tiny).to(dtype=badge_embeddings.dtype)
        weighted_embeddings = badge_embeddings * weights.unsqueeze(1)
        norm_time = time.perf_counter() - t_norm

        badge_norms = badge_embeddings.float().norm(dim=1)
        weighted_norms = weighted_embeddings.float().norm(dim=1)
        raw_stats = _tensor_stats(raw_adv_scores)
        norm_stats = _tensor_stats(norm_adv_scores)
        badge_norm_stats = _tensor_stats(badge_norms)
        weighted_norm_stats = _tensor_stats(weighted_norms)

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[BADGE_ADV_MULT] raw_adv_stats "
                    f"min={raw_stats['min']:.6f} max={raw_stats['max']:.6f} "
                    f"mean={raw_stats['mean']:.6f} std={raw_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_MULT] norm_adv_stats "
                    f"min={norm_stats['min']:.6f} max={norm_stats['max']:.6f} "
                    f"mean={norm_stats['mean']:.6f} std={norm_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_MULT] badge_norm_stats "
                    f"min={badge_norm_stats['min']:.6f} max={badge_norm_stats['max']:.6f} "
                    f"mean={badge_norm_stats['mean']:.6f} std={badge_norm_stats['std']:.6f}"
                ),
                device=str(device),
            )
            progress_logger.log(
                (
                    "[BADGE_ADV_MULT] weighted_norm_stats "
                    f"min={weighted_norm_stats['min']:.6f} max={weighted_norm_stats['max']:.6f} "
                    f"mean={weighted_norm_stats['mean']:.6f} std={weighted_norm_stats['std']:.6f}"
                ),
                device=str(device),
            )

        candidate_local = np.arange(weighted_embeddings.size(0), dtype=np.int64)
        if self.cfg.badge_candidate_cap is not None and self.cfg.badge_candidate_cap < weighted_embeddings.size(0):
            rng = np.random.default_rng(self.cfg.seed)
            candidate_local = rng.choice(candidate_local, size=self.cfg.badge_candidate_cap, replace=False)
            weighted_for_selection = weighted_embeddings[candidate_local]
        else:
            weighted_for_selection = weighted_embeddings

        t_kmeans = time.perf_counter()
        picked_local_in_candidates = select_badge_kmeanspp(weighted_for_selection, B=budget, seed=self.cfg.seed)
        picked_local = candidate_local[picked_local_in_candidates]
        selected = unlabeled_indices[picked_local]
        kmeans_time = time.perf_counter() - t_kmeans

        total_time = time.perf_counter() - total_t0
        scoring_time = badge_time + adv_time + norm_time
        selection_time = kmeans_time

        if progress_logger is not None:
            progress_logger.log(
                (
                    "[BADGE_ADV_MULT] timing "
                    f"badge_embed={badge_time:.3f}s adv_scores={adv_time:.3f}s "
                    f"normalization={norm_time:.3f}s kmeanspp={kmeans_time:.3f}s "
                    f"total={total_time:.3f}s"
                ),
                device=str(device),
            )

        debug_data = None
        if self.cfg.debug_save_adv_scores:
            debug_data = {
                "__file_tag": "adv_scores",
                "__column_order": [
                    "sample_index",
                    "raw_adv_score",
                    "normalized_adv_score",
                    "badge_embedding_norm",
                    "weighted_embedding_norm",
                ],
                "sample_index": np.asarray(unlabeled_indices, dtype=np.int64),
                "raw_adv_score": raw_adv_scores.numpy(),
                "normalized_adv_score": norm_adv_scores.numpy(),
                "badge_embedding_norm": badge_norms.numpy(),
                "weighted_embedding_norm": weighted_norms.numpy(),
            }

        return AcquisitionOutput(
            selected_indices=selected,
            scores=None,
            scoring_time_sec=scoring_time,
            selection_time_sec=selection_time,
            extras={
                "embedding_dim": int(badge_embeddings.size(1)),
                "projection_dim": int(self.cfg.badge_projection_dim),
                "candidate_cap": self.cfg.badge_candidate_cap,
                "epsilon_acq": float(self.cfg.epsilon_acq),
                "attack_type": self.cfg.adv_attack_type_for_acquisition,
                "pgd_steps": int(self.cfg.adv_pgd_steps),
                "pgd_step_size": self.cfg.adv_pgd_step_size,
                "adv_score_normalization": self.cfg.adv_score_normalization,
                "adv_tiny": float(self.cfg.adv_tiny),
                "adv_raw_stats": raw_stats,
                "adv_normalized_stats": norm_stats,
                "badge_norm_stats": badge_norm_stats,
                "weighted_norm_stats": weighted_norm_stats,
                "timing_badge_embeddings_sec": float(badge_time),
                "timing_adv_scores_sec": float(adv_time),
                "timing_normalization_sec": float(norm_time),
                "timing_kmeanspp_sec": float(kmeans_time),
                "timing_total_acquisition_sec": float(total_time),
            },
            debug_data=debug_data,
        )

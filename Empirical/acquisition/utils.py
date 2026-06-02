import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


@dataclass
class AcquisitionOutput:
    selected_indices: np.ndarray
    scores: Optional[np.ndarray]
    scoring_time_sec: float
    selection_time_sec: float
    extras: Dict[str, Any]
    debug_data: Optional[Dict[str, Any]] = None


class BaseAcquisition(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
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
        raise NotImplementedError


def topk_unlabeled_indices(
    unlabeled_indices: np.ndarray,
    scores: torch.Tensor,
    budget: int,
) -> np.ndarray:
    k = min(budget, scores.numel())
    top_local = torch.topk(scores, k=k, largest=True).indices.cpu().numpy()
    return unlabeled_indices[top_local]


def choose_without_replacement(indices: np.ndarray, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(budget, len(indices))
    picked = rng.choice(indices, size=k, replace=False)
    return np.asarray(picked, dtype=np.int64)


def timing_wrapper(fn, *args, **kwargs):
    start = time.perf_counter()
    output = fn(*args, **kwargs)
    return output, time.perf_counter() - start


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    xf = x.float()
    return {
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
    }


def _channel_tensor(values: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def scaled_linf_eps(
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


def clamp_to_valid_range(
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


def compute_logit_mismatch_scores(
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
    progress_method_name: str = "C_SCORE",
) -> torch.Tensor:
    """
    Compute per-sample robustness score:
      c(x) = max_{||delta||_inf <= epsilon_acq} || z(x + delta) - z(x) ||_2^2

    Implementation details:
    - clean branch is detached: z_clean = model(x).detach()
    - optimize only delta via gradient ascent
    - objective uses logits mismatch ONLY (no CE / labels)
    """
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
        x0 = images.to(device, non_blocking=True).detach()
        channels = x0.size(1)
        eps_t = scaled_linf_eps(
            epsilon=epsilon_acq,
            std=std,
            device=x0.device,
            dtype=x0.dtype,
            channels=channels,
        )

        # Mandatory detached clean branch.
        with torch.no_grad():
            z_clean = model(x0).detach()

        if attack_type == "fgsm":
            # One ascent step on delta at delta=0.
            delta = torch.zeros_like(x0, requires_grad=True)
            z_adv = model(clamp_to_valid_range(x0 + delta, mean=mean, std=std))
            mismatch_obj = (z_adv - z_clean).pow(2).sum(dim=1).mean()
            grad_delta = torch.autograd.grad(mismatch_obj, delta, only_inputs=True)[0]
            delta = torch.clamp(eps_t * grad_delta.sign(), min=-eps_t, max=eps_t).detach()
            x_adv = clamp_to_valid_range(x0 + delta, mean=mean, std=std).detach()
        else:
            steps = max(1, int(pgd_steps))
            step_size = float(pgd_step_size) if pgd_step_size is not None else float(epsilon_acq) / float(steps)
            alpha_t = scaled_linf_eps(
                epsilon=step_size,
                std=std,
                device=x0.device,
                dtype=x0.dtype,
                channels=channels,
            )

            # Random init inside epsilon-ball.
            delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
            delta = clamp_to_valid_range(x0 + delta, mean=mean, std=std) - x0

            for _ in range(steps):
                delta = delta.detach().requires_grad_(True)
                z_adv = model(clamp_to_valid_range(x0 + delta, mean=mean, std=std))
                mismatch_obj = (z_adv - z_clean).pow(2).sum(dim=1).mean()
                grad_delta = torch.autograd.grad(mismatch_obj, delta, only_inputs=True)[0]
                delta = delta.detach() + alpha_t * grad_delta.sign()
                delta = torch.clamp(delta, min=-eps_t, max=eps_t)
                delta = clamp_to_valid_range(x0 + delta, mean=mean, std=std) - x0

            x_adv = clamp_to_valid_range(x0 + delta.detach(), mean=mean, std=std).detach()

        with torch.no_grad():
            z_adv_final = model(x_adv)
            c_scores = (z_adv_final - z_clean).pow(2).sum(dim=1)
            all_scores.append(c_scores.cpu())

        if progress_logger is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
            progress_logger.log_scoring_eta(
                method=progress_method_name,
                processed_batches=batch_idx,
                total_batches=total_batches,
                elapsed=time.perf_counter() - t0,
                device=str(device),
            )

    return torch.cat(all_scores, dim=0)

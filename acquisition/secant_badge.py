import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from acquisition.badge import (
    last_layer_gradient_embedding_from_logits_features,
    select_badge_kmeanspp,
)
from acquisition.logdet_adv_disp import _fgsm_predictive_ce_attack, _pgd_predictive_ce_attack, forward_with_features
from acquisition.utils import AcquisitionOutput, BaseAcquisition, scaled_linf_eps, tensor_stats


def _secant_attack_settings(cfg) -> Tuple[str, int, Optional[float], bool]:
    attack_type = str(getattr(cfg, "acquisition_attack", "pgd")).lower()
    if attack_type not in {"fgsm", "pgd"}:
        attack_type = str(getattr(cfg, "adv_attack_type_for_acquisition", "pgd")).lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported robust BADGE attack type: {attack_type}")

    steps = int(getattr(cfg, "attack_steps", getattr(cfg, "acquisition_pgd_steps", 3)))
    if steps <= 0:
        raise ValueError(f"robust BADGE attack steps must be positive, got {steps}")
    step_size = getattr(cfg, "attack_step_size", None)
    random_start = bool(getattr(cfg, "attack_random_start", True))
    return attack_type, steps, step_size, random_start


def _maybe_project_embeddings(
    embeddings: torch.Tensor,
    projection_dim: int,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    input_dim = int(embeddings.size(1))
    if projection_dim <= 0 or projection_dim >= input_dim:
        return embeddings, 0
    g = torch.Generator(device=embeddings.device)
    g.manual_seed(int(seed))
    proj = torch.randn(
        input_dim,
        int(projection_dim),
        generator=g,
        device=embeddings.device,
        dtype=embeddings.dtype,
    ) / math.sqrt(float(projection_dim))
    return embeddings @ proj, int(projection_dim)


def build_secant_badge_embeddings(g_clean: torch.Tensor, g_adv: torch.Tensor) -> torch.Tensor:
    """Build psi_secant(x) = [g_clean(x); g_adv(x) - g_clean(x)]."""
    if g_clean.shape != g_adv.shape:
        raise ValueError(f"g_clean and g_adv must have identical shapes, got {g_clean.shape} vs {g_adv.shape}")
    return torch.cat([g_clean, g_adv - g_clean], dim=1)


def build_jointadv_badge_embeddings(g_clean: torch.Tensor, g_adv: torch.Tensor) -> torch.Tensor:
    """Build psi_jointadv(x) = [g_clean(x); g_adv(x)]."""
    if g_clean.shape != g_adv.shape:
        raise ValueError(f"g_clean and g_adv must have identical shapes, got {g_clean.shape} vs {g_adv.shape}")
    return torch.cat([g_clean, g_adv], dim=1)


def compute_clean_grad_norm(g_clean: torch.Tensor) -> torch.Tensor:
    """Compute the secant clean-anchor filter score ||g_clean(x)||_2."""
    if g_clean.ndim != 2:
        raise ValueError(f"g_clean must be [N,D], got shape={tuple(g_clean.shape)}")
    return g_clean.float().norm(dim=1)


def compute_joint_embedding_norm(g_clean: torch.Tensor, g_adv: torch.Tensor) -> torch.Tensor:
    """Compute the jointadv filter score ||[g_clean(x); g_adv(x)]||_2."""
    if g_clean.ndim != 2 or g_adv.ndim != 2:
        raise ValueError(f"g_clean and g_adv must be [N,D], got {tuple(g_clean.shape)} and {tuple(g_adv.shape)}")
    if g_clean.shape != g_adv.shape:
        raise ValueError(f"g_clean and g_adv must have identical shapes, got {g_clean.shape} vs {g_adv.shape}")
    g_clean_f = g_clean.float()
    g_adv_f = g_adv.float()
    return torch.sqrt(g_clean_f.pow(2).sum(dim=1) + g_adv_f.pow(2).sum(dim=1))


def _scalar_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    xf = x.float().detach().cpu()
    return {
        "min": float(xf.min().item()),
        "median": float(torch.median(xf).item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
    }


def apply_percentile_prefilter(
    metric: torch.Tensor,
    drop_percent: float,
    candidate_indices: np.ndarray,
    budget: int,
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Drop the bottom p% by metric and preserve the original pool order.

    p10 keeps ceil(0.9N), while always keeping at least the acquisition budget.
    Ties are deterministic: higher metric first, then lower original position.
    """
    if metric.ndim != 1:
        raise ValueError(f"prefilter metric must be [N], got shape={tuple(metric.shape)}")
    if not (0.0 <= float(drop_percent) <= 100.0):
        raise ValueError(f"drop_percent must be in [0,100], got {drop_percent}")
    n = int(metric.numel())
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidate_indices) != n:
        raise ValueError(f"candidate_indices length must match metric length, got {len(candidate_indices)} vs {n}")

    scores = metric.float()
    if n == 0:
        empty = np.array([], dtype=np.int64)
        stats = _scalar_stats(scores)
        return empty, empty, {
            "enabled": False,
            "metric": metric_name,
            "drop_percent": float(drop_percent),
            "pool_size_before": 0,
            "pool_size_after": 0,
            "drop_count": 0,
            "keep_count": 0,
            "threshold": float("nan"),
            "score_stats": stats,
            "retained_score_stats": stats,
        }

    if float(drop_percent) <= 0.0:
        retained_local = np.arange(n, dtype=np.int64)
        stats = _scalar_stats(scores)
        return retained_local, candidate_indices, {
            "enabled": False,
            "metric": metric_name,
            "drop_percent": 0.0,
            "pool_size_before": n,
            "pool_size_after": n,
            "drop_count": 0,
            "keep_count": n,
            "threshold": float("-inf"),
            "score_stats": stats,
            "retained_score_stats": stats,
        }

    keep_count = int(np.ceil((1.0 - float(drop_percent) / 100.0) * float(n)))
    keep_count = max(int(budget), keep_count)
    keep_count = max(1, min(keep_count, n))

    scores_np = scores.detach().cpu().numpy()
    ranked = np.lexsort((np.arange(n, dtype=np.int64), -scores_np))
    retained_unsorted = ranked[:keep_count].astype(np.int64)
    retained_local = np.sort(retained_unsorted).astype(np.int64)
    retained_scores = scores[torch.as_tensor(retained_local, dtype=torch.long, device=scores.device)]
    threshold = float(np.min(scores_np[retained_unsorted])) if keep_count > 0 else float("nan")

    return retained_local, candidate_indices[retained_local], {
        "enabled": True,
        "metric": metric_name,
        "drop_percent": float(drop_percent),
        "pool_size_before": n,
        "pool_size_after": int(keep_count),
        "drop_count": int(n - keep_count),
        "keep_count": int(keep_count),
        "threshold": threshold,
        "score_stats": _scalar_stats(scores),
        "retained_score_stats": _scalar_stats(retained_scores),
    }


def _normalize_embedding_type(embedding_type: str) -> str:
    kind = str(embedding_type).lower()
    if kind in {"secant", "ours_badge_secant", "ours_secant_badge"}:
        return "secant"
    if kind in {"jointadv", "joint_adv", "ours_badge_jointadv"}:
        return "jointadv"
    raise ValueError(f"Unsupported robust BADGE embedding type: {embedding_type}")


def _build_robust_badge_embeddings(g_clean: torch.Tensor, g_adv: torch.Tensor, embedding_type: str) -> torch.Tensor:
    kind = _normalize_embedding_type(embedding_type)
    if kind == "secant":
        return build_secant_badge_embeddings(g_clean=g_clean, g_adv=g_adv)
    return build_jointadv_badge_embeddings(g_clean=g_clean, g_adv=g_adv)


def compute_adv_last_layer_grad_embeddings(
    model,
    x0: torch.Tensor,
    pseudo_labels: torch.Tensor,
    epsilon_acq: float,
    attack_type: str,
    attack_steps: int,
    attack_step_size: Optional[float],
    attack_random_start: bool,
    mean=None,
    std=None,
) -> torch.Tensor:
    """
    Compute adversarial last-layer BADGE gradients for one batch.

    The clean pseudo-labels are fixed during the acquisition-time CE attack and
    reused when forming the analytic last-layer gradient on x_adv.
    """
    if float(epsilon_acq) <= 0.0:
        with torch.no_grad():
            adv_features, adv_logits = forward_with_features(model=model, x=x0, require_features=True)
            return last_layer_gradient_embedding_from_logits_features(
                logits=adv_logits,
                features=adv_features,
                pseudo_labels=pseudo_labels,
            )

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
        return last_layer_gradient_embedding_from_logits_features(
            logits=adv_logits,
            features=adv_features,
            pseudo_labels=pseudo_labels,
        )


def compute_robust_badge_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    epsilon_acq: float = 1.0 / 255.0,
    attack_type: str = "pgd",
    attack_steps: int = 3,
    attack_step_size: Optional[float] = None,
    attack_random_start: bool = True,
    mean=None,
    std=None,
    projection_dim: int = 0,
    seed: int = 0,
    progress_logger=None,
    return_parts: bool = False,
    embedding_type: str = "secant",
    progress_method_name: Optional[str] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Build a robust BADGE embedding in analytic last-layer gradient space.

    embedding_type="secant":
      psi_secant(x) = [g_clean(x); g_adv(x) - g_clean(x)]

    embedding_type="jointadv":
      psi_jointadv(x) = [g_clean(x); g_adv(x)]

    The clean pseudo-label y_hat is fixed during the CE attack and when building
    both clean and adversarial last-layer gradient embeddings.
    """
    if device is None:
        device = next(model.parameters()).device
    if int(attack_steps) <= 0:
        raise ValueError(f"attack_steps must be positive, got {attack_steps}")
    attack_type = str(attack_type).lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type: {attack_type}")
    embedding_type = _normalize_embedding_type(embedding_type)
    if progress_method_name is None:
        progress_method_name = "OURS_BADGE_SECANT" if embedding_type == "secant" else "OURS_BADGE_JOINTADV"

    was_training = model.training
    model.eval()

    phi_all = []
    g_clean_all = [] if return_parts else None
    g_adv_all = [] if return_parts else None
    correction_all = [] if return_parts else None
    clean_norms = []
    adv_norms = []
    correction_norms = []

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
            g_adv = g_clean
        else:
            g_adv = compute_adv_last_layer_grad_embeddings(
                model=model,
                x0=x0,
                pseudo_labels=pseudo_labels,
                epsilon_acq=float(epsilon_acq),
                attack_type=attack_type,
                attack_steps=int(attack_steps),
                attack_step_size=attack_step_size,
                attack_random_start=bool(attack_random_start),
                mean=mean,
                std=std,
            )

        with torch.no_grad():
            correction = g_adv - g_clean
            phi = _build_robust_badge_embeddings(
                g_clean=g_clean,
                g_adv=g_adv,
                embedding_type=embedding_type,
            )
            phi, used_projection_dim = _maybe_project_embeddings(
                embeddings=phi,
                projection_dim=int(projection_dim),
                seed=int(seed),
            )

            clean_norms.append(g_clean.norm(dim=1).detach().cpu())
            adv_norms.append(g_adv.norm(dim=1).detach().cpu())
            correction_norms.append(correction.norm(dim=1).detach().cpu())
            phi_all.append(phi.detach().cpu())
            if return_parts:
                g_clean_all.append(g_clean.detach().cpu())
                g_adv_all.append(g_adv.detach().cpu())
                correction_all.append(correction.detach().cpu())

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

    if len(phi_all) == 0:
        empty = torch.empty((0, 0), dtype=torch.float32)
        if return_parts:
            return empty, {
                "g_clean": empty,
                "g_adv": empty,
                "correction": empty,
                "clean_norm": torch.empty((0,), dtype=torch.float32),
                "adv_norm": torch.empty((0,), dtype=torch.float32),
                "correction_norm": torch.empty((0,), dtype=torch.float32),
                "projection_dim": torch.tensor(0),
            }
        return empty

    phi_cat = torch.cat(phi_all, dim=0)
    parts: Dict[str, Any] = {
        "clean_norm": torch.cat(clean_norms, dim=0),
        "adv_norm": torch.cat(adv_norms, dim=0),
        "correction_norm": torch.cat(correction_norms, dim=0),
        "projection_dim": torch.tensor(int(used_projection_dim)),
        "embedding_type": embedding_type,
    }
    if return_parts:
        parts.update(
            {
                "g_clean": torch.cat(g_clean_all, dim=0),
                "g_adv": torch.cat(g_adv_all, dim=0),
                "correction": torch.cat(correction_all, dim=0),
            }
        )
        return phi_cat, parts
    return phi_cat


def compute_secant_badge_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    epsilon_acq: float = 1.0 / 255.0,
    attack_type: str = "pgd",
    attack_steps: int = 3,
    attack_step_size: Optional[float] = None,
    attack_random_start: bool = True,
    mean=None,
    std=None,
    projection_dim: int = 0,
    seed: int = 0,
    progress_logger=None,
    return_parts: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Build psi_secant(x) = [g_clean(x); g_adv(x) - g_clean(x)]."""
    return compute_robust_badge_embeddings(
        model=model,
        unlabeled_loader=unlabeled_loader,
        device=device,
        epsilon_acq=epsilon_acq,
        attack_type=attack_type,
        attack_steps=attack_steps,
        attack_step_size=attack_step_size,
        attack_random_start=attack_random_start,
        mean=mean,
        std=std,
        projection_dim=projection_dim,
        seed=seed,
        progress_logger=progress_logger,
        return_parts=return_parts,
        embedding_type="secant",
        progress_method_name="OURS_SECANT_BADGE",
    )


def compute_jointadv_badge_embeddings(
    model,
    unlabeled_loader,
    device: Optional[torch.device] = None,
    epsilon_acq: float = 1.0 / 255.0,
    attack_type: str = "pgd",
    attack_steps: int = 3,
    attack_step_size: Optional[float] = None,
    attack_random_start: bool = True,
    mean=None,
    std=None,
    projection_dim: int = 0,
    seed: int = 0,
    progress_logger=None,
    return_parts: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Build psi_jointadv(x) = [g_clean(x); g_adv(x)]."""
    return compute_robust_badge_embeddings(
        model=model,
        unlabeled_loader=unlabeled_loader,
        device=device,
        epsilon_acq=epsilon_acq,
        attack_type=attack_type,
        attack_steps=attack_steps,
        attack_step_size=attack_step_size,
        attack_random_start=attack_random_start,
        mean=mean,
        std=std,
        projection_dim=projection_dim,
        seed=seed,
        progress_logger=progress_logger,
        return_parts=return_parts,
        embedding_type="jointadv",
        progress_method_name="OURS_BADGE_JOINTADV",
    )


def run_badge_selection_from_embeddings(
    embeddings: torch.Tensor,
    unlabeled_indices: np.ndarray,
    budget: int,
    seed: int,
    candidate_cap: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Run the repository's BADGE k-means++ selector on precomputed embeddings."""
    candidate_local = np.arange(embeddings.size(0), dtype=np.int64)
    if candidate_cap is not None and int(candidate_cap) < embeddings.size(0):
        rng = np.random.default_rng(int(seed))
        candidate_local = rng.choice(candidate_local, size=int(candidate_cap), replace=False)
        embeddings_sel = embeddings[candidate_local]
    else:
        embeddings_sel = embeddings

    t_select = time.perf_counter()
    picked_local_in_candidates = select_badge_kmeanspp(embeddings_sel, B=budget, seed=int(seed))
    picked_local = candidate_local[picked_local_in_candidates]
    selected = unlabeled_indices[picked_local]
    selection_time = time.perf_counter() - t_select
    return np.asarray(selected, dtype=np.int64), float(selection_time), picked_local


class _RobustBADGEStrategy(BaseAcquisition):
    method_name = "ours_badge_secant"
    embedding_type = "secant"
    embedding_description = "concat_clean_last_layer_gradient_and_adv_minus_clean_gradient"
    progress_method_name = "OURS_BADGE_SECANT"
    default_prefilter_metric = "secant_clean_grad_norm"

    def _resolve_prefilter_metric(self, drop_percent: float) -> str:
        raw = str(getattr(self.cfg, "prefilter_metric", "none")).lower()
        if raw in {"", "none", "off", "false"}:
            return "none" if float(drop_percent) <= 0.0 else self.default_prefilter_metric

        if self.embedding_type == "secant":
            if raw in {
                "secant_clean_grad_norm",
                "secant_clean_gradient_norm",
                "clean_grad_norm",
                "clean_gradient_norm",
                "g_clean_norm",
                "d_clean",
            }:
                return "secant_clean_grad_norm"
            raise ValueError(
                f"{self.method_name} supports only secant_clean_grad_norm prefiltering; got {raw}"
            )

        if raw in {"joint_embedding_norm", "joint_norm", "jointadv_norm", "psi_jointadv_norm"}:
            return "joint_embedding_norm"
        raise ValueError(f"{self.method_name} supports only joint_embedding_norm prefiltering; got {raw}")

    def _compute_prefilter_scores(self, parts: Dict[str, torch.Tensor], metric: str) -> torch.Tensor:
        if metric == "secant_clean_grad_norm":
            return compute_clean_grad_norm(parts["g_clean"])
        if metric == "joint_embedding_norm":
            return compute_joint_embedding_norm(parts["g_clean"], parts["g_adv"])
        raise ValueError(f"Unsupported robust BADGE prefilter metric: {metric}")

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
        if progress_logger is not None:
            progress_logger.log(
                f"[acq] method={self.method_name} embedding={self.embedding_type}",
                device=str(device),
            )

        t0 = time.perf_counter()
        embeddings, parts = compute_robust_badge_embeddings(
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
            projection_dim=int(getattr(self.cfg, "badge_projection_dim", 0)),
            seed=int(self.cfg.seed),
            progress_logger=progress_logger,
            return_parts=True,
            embedding_type=self.embedding_type,
            progress_method_name=self.progress_method_name,
        )
        scoring_time = time.perf_counter() - t0

        prefilter_drop_percent = float(getattr(self.cfg, "prefilter_drop_percent", 0.0))
        prefilter_metric = self._resolve_prefilter_metric(prefilter_drop_percent)
        prefilter_scores = None
        retained_local = np.arange(embeddings.size(0), dtype=np.int64)
        embeddings_for_selection = embeddings
        unlabeled_for_selection = np.asarray(unlabeled_indices, dtype=np.int64)
        prefilter_debug = {
            "enabled": False,
            "metric": "none",
            "drop_percent": 0.0,
            "pool_size_before": int(embeddings.size(0)),
            "pool_size_after": int(embeddings.size(0)),
            "drop_count": 0,
            "keep_count": int(embeddings.size(0)),
            "threshold": float("-inf"),
            "score_stats": {},
            "retained_score_stats": {},
        }
        if prefilter_metric != "none":
            prefilter_scores = self._compute_prefilter_scores(parts=parts, metric=prefilter_metric)
            retained_local, unlabeled_for_selection, prefilter_debug = apply_percentile_prefilter(
                metric=prefilter_scores,
                drop_percent=prefilter_drop_percent,
                candidate_indices=unlabeled_indices,
                budget=budget,
                metric_name=prefilter_metric,
            )
            if bool(prefilter_debug["enabled"]):
                retained_t = torch.as_tensor(retained_local, dtype=torch.long, device=embeddings.device)
                embeddings_for_selection = embeddings[retained_t]
                if progress_logger is not None:
                    stats = prefilter_debug["score_stats"]
                    progress_logger.log(
                        (
                            f"[prefilter] method={self.method_name} "
                            f"metric={prefilter_metric} "
                            f"drop_percent={prefilter_drop_percent:.3f} "
                            f"pool={prefilter_debug['pool_size_before']}->{prefilter_debug['pool_size_after']} "
                            f"min={stats['min']:.6f} "
                            f"median={stats['median']:.6f} "
                            f"max={stats['max']:.6f}"
                        ),
                        device=str(device),
                    )

        candidate_cap = getattr(self.cfg, "badge_candidate_cap", None)
        selected, selection_time, picked_local_filtered = run_badge_selection_from_embeddings(
            embeddings=embeddings_for_selection,
            unlabeled_indices=unlabeled_for_selection,
            budget=budget,
            seed=int(self.cfg.seed),
            candidate_cap=candidate_cap,
        )
        picked_local = retained_local[picked_local_filtered]

        selected_prefilter_mean = float("nan")
        if prefilter_scores is not None and len(picked_local) > 0:
            selected_t = torch.as_tensor(picked_local, dtype=torch.long, device=prefilter_scores.device)
            selected_prefilter_mean = float(prefilter_scores[selected_t].float().mean().item())

        clean_stats = tensor_stats(parts["clean_norm"])
        adv_stats = tensor_stats(parts["adv_norm"])
        correction_stats = tensor_stats(parts["correction_norm"])
        if progress_logger is not None:
            progress_logger.log(
                (
                    f"[acq] method={self.method_name} norm_stats "
                    f"clean_mean={clean_stats['mean']:.6f} "
                    f"adv_mean={adv_stats['mean']:.6f} "
                    f"correction_mean={correction_stats['mean']:.6f} "
                    f"selected={int(len(selected))}"
                ),
                device=str(device),
            )
            if bool(prefilter_debug["enabled"]):
                progress_logger.log(
                    (
                        f"[prefilter] method={self.method_name} "
                        f"metric={prefilter_metric} selected_mean={selected_prefilter_mean:.6f}"
                    ),
                    device=str(device),
                )

        return AcquisitionOutput(
            selected_indices=np.asarray(selected, dtype=np.int64),
            scores=None,
            scoring_time_sec=float(scoring_time),
            selection_time_sec=float(selection_time),
            extras={
                "method": self.method_name,
                "embedding": self.embedding_description,
                "embedding_type": self.embedding_type,
                "embedding_dim": int(embeddings.size(1)),
                "base_gradient_dim": int(parts["g_clean"].size(1)),
                "projection_dim": int(parts["projection_dim"].item()),
                "candidate_cap": candidate_cap,
                "prefilter_enabled": bool(prefilter_debug["enabled"]),
                "prefilter_metric": prefilter_debug["metric"] if bool(prefilter_debug["enabled"]) else "none",
                "prefilter_drop_percent": float(prefilter_debug["drop_percent"]),
                "prefilter_pool_size_before": int(prefilter_debug["pool_size_before"]),
                "prefilter_pool_size_after": int(prefilter_debug["pool_size_after"]),
                "prefilter_drop_count": int(prefilter_debug["drop_count"]),
                "prefilter_keep_count": int(prefilter_debug["keep_count"]),
                "prefilter_threshold": float(prefilter_debug["threshold"]),
                "prefilter_score_stats": prefilter_debug["score_stats"],
                "prefilter_retained_score_stats": prefilter_debug["retained_score_stats"],
                "prefilter_selected_mean_score": selected_prefilter_mean,
                "attack_type": attack_type,
                "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
                "attack_steps": int(attack_steps),
                "attack_step_size": attack_step_size,
                "attack_random_start": bool(attack_random_start),
                "gradient_embedding": "last_layer_weight_gradient_no_bias",
                "g_clean_norm_stats": clean_stats,
                "g_adv_norm_stats": adv_stats,
                "correction_norm_stats": correction_stats,
            },
        )


class OursSecantBADGEStrategy(_RobustBADGEStrategy):
    method_name = "ours_secant_badge"
    embedding_type = "secant"
    embedding_description = "concat_clean_last_layer_gradient_and_adv_minus_clean_gradient"
    progress_method_name = "OURS_SECANT_BADGE"
    default_prefilter_metric = "secant_clean_grad_norm"


class OursBadgeSecantStrategy(_RobustBADGEStrategy):
    method_name = "ours_badge_secant"
    embedding_type = "secant"
    embedding_description = "concat_clean_last_layer_gradient_and_adv_minus_clean_gradient"
    progress_method_name = "OURS_BADGE_SECANT"
    default_prefilter_metric = "secant_clean_grad_norm"


class OursBadgeJointAdvStrategy(_RobustBADGEStrategy):
    method_name = "ours_badge_jointadv"
    embedding_type = "jointadv"
    embedding_description = "concat_clean_last_layer_gradient_and_adv_gradient"
    progress_method_name = "OURS_BADGE_JOINTADV"
    default_prefilter_metric = "joint_embedding_norm"

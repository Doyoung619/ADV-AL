import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from acquisition.badge import last_layer_gradient_embedding_from_logits_features, select_badge_kmeanspp
from acquisition.logdet_adv_disp import _fgsm_predictive_ce_attack, _pgd_predictive_ce_attack, forward_with_features
from acquisition.utils import AcquisitionOutput, BaseAcquisition, scaled_linf_eps, tensor_stats


def _secant_attack_settings(cfg) -> Tuple[str, int, Optional[float], bool]:
    attack_type = str(getattr(cfg, "acquisition_attack", "pgd")).lower()
    if attack_type not in {"fgsm", "pgd"}:
        attack_type = str(getattr(cfg, "adv_attack_type_for_acquisition", "pgd")).lower()
    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported ours_secant_badge attack type: {attack_type}")

    steps = int(getattr(cfg, "attack_steps", getattr(cfg, "acquisition_pgd_steps", 3)))
    if steps <= 0:
        raise ValueError(f"ours_secant_badge attack steps must be positive, got {steps}")
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
    """
    Build phi(x) = [g_clean(x); g_adv(x) - g_clean(x)] in BADGE gradient space.

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
            correction = g_adv - g_clean
            phi = torch.cat([g_clean, correction], dim=1)
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
                method="OURS_SECANT_BADGE",
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


class OursSecantBADGEStrategy(BaseAcquisition):
    method_name = "ours_secant_badge"

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
        t0 = time.perf_counter()
        embeddings, parts = compute_secant_badge_embeddings(
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
        )
        scoring_time = time.perf_counter() - t0

        candidate_local = np.arange(embeddings.size(0), dtype=np.int64)
        candidate_cap = getattr(self.cfg, "badge_candidate_cap", None)
        if candidate_cap is not None and int(candidate_cap) < embeddings.size(0):
            rng = np.random.default_rng(int(self.cfg.seed))
            candidate_local = rng.choice(candidate_local, size=int(candidate_cap), replace=False)
            embeddings_sel = embeddings[candidate_local]
        else:
            embeddings_sel = embeddings

        t1 = time.perf_counter()
        picked_local_in_candidates = select_badge_kmeanspp(embeddings_sel, B=budget, seed=int(self.cfg.seed))
        picked_local = candidate_local[picked_local_in_candidates]
        selected = unlabeled_indices[picked_local]
        selection_time = time.perf_counter() - t1

        clean_stats = tensor_stats(parts["clean_norm"])
        adv_stats = tensor_stats(parts["adv_norm"])
        correction_stats = tensor_stats(parts["correction_norm"])
        if progress_logger is not None:
            progress_logger.log(
                (
                    "[OURS_SECANT_BADGE] norm_stats "
                    f"clean_mean={clean_stats['mean']:.6f} "
                    f"adv_mean={adv_stats['mean']:.6f} "
                    f"correction_mean={correction_stats['mean']:.6f} "
                    f"selected={int(len(selected))}"
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
                "embedding": "concat_clean_last_layer_gradient_and_adv_minus_clean_gradient",
                "embedding_dim": int(embeddings.size(1)),
                "base_gradient_dim": int(parts["g_clean"].size(1)),
                "projection_dim": int(parts["projection_dim"].item()),
                "candidate_cap": candidate_cap,
                "attack_type": attack_type,
                "epsilon_acq": float(getattr(self.cfg, "epsilon_acq", 1.0 / 255.0)),
                "attack_steps": int(attack_steps),
                "attack_step_size": attack_step_size,
                "attack_random_start": bool(attack_random_start),
                "g_clean_norm_stats": clean_stats,
                "g_adv_norm_stats": adv_stats,
                "correction_norm_stats": correction_stats,
            },
        )

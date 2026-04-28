"""Active-learning loop driver for the GLM theory experiments.

Each method (random / badge / secant_badge / oracle_logdet) maintains its own copy of
(model, labeled-set indices, unlabeled-pool indices) so that comparisons are fair under a
shared initial split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from glm_theory.acquisition import (
    badge_acquire,
    oracle_logdet_acquire,
    random_acquire,
    secant_badge_acquire,
)
from glm_theory.adversarial import fgsm_perturb
from glm_theory.contraction import contraction_probe
from glm_theory.glms import BaseGLM, GLM_REGISTRY, GaussianGLM, PoissonGLM
from glm_theory.theory_metrics import (
    clean_gram,
    condition_number,
    lambda_min,
    logdet,
    min_positive_eig,
    secant_gram,
)


log = logging.getLogger(__name__)


@dataclass
class ALConfig:
    glm_family: str
    d: int = 20
    C: int = 5
    sigma: float = 1.0
    n_init: int = 100
    pool_size: int = 2000
    q: int = 50
    n_rounds: int = 20
    alpha: float = 75.0
    eps_acq: float = 0.25
    methods: List[str] = field(
        default_factory=lambda: ["random", "badge", "secant_badge"]
    )
    contraction_K: int = 100
    contraction_lr: float = 0.01
    ridge: float = 1e-6
    fit_ridge: float = 1e-4
    w_scale: float = 1.0
    pseudo_jitter: float = 0.1  # additive noise on regression pseudo-targets
    decay_curves_seed: int = 0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


def _build_glm(cfg: ALConfig) -> BaseGLM:
    cls = GLM_REGISTRY[cfg.glm_family]
    return cls(d=cfg.d, C=cfg.C, sigma=cfg.sigma, ridge=cfg.fit_ridge, w_scale=cfg.w_scale)


def _pseudo_targets(glm: BaseGLM, theta: np.ndarray, X: np.ndarray,
                    jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Pseudo-labels for acquisition.

    Classification uses argmax / threshold predictions. Regression GLMs add a small frozen
    jitter so that the clean loss-gradient does not vanish identically when the pseudo-target
    equals the model's own prediction (see paper's note in §Adversarial perturbation).
    """
    y = glm.pseudo_label(theta, X)
    if isinstance(glm, (GaussianGLM, PoissonGLM)) and jitter > 0.0:
        y = y + rng.normal(scale=jitter, size=y.shape)
    return y


def _select(method: str, *, g_clean: np.ndarray, g_adv: np.ndarray,
            alpha: float, q: int, rng: np.random.Generator) -> np.ndarray:
    if method == "random":
        return random_acquire(rng, g_clean.shape[0], q)
    if method == "badge":
        return badge_acquire(g_clean, q, rng)
    if method == "secant_badge":
        return secant_badge_acquire(g_clean, g_adv, alpha, q, rng)
    if method == "oracle_logdet":
        phi = np.concatenate([g_clean, g_adv - g_clean], axis=1)
        return oracle_logdet_acquire(phi, q)
    raise ValueError(f"unknown method '{method}'")


def run_one_seed(cfg: ALConfig, seed: int) -> List[Dict]:
    """Run the AL loop for a single seed across all configured methods.

    Returns one dict per (method, round). Also returns decay-curve rows when the seed matches
    ``cfg.decay_curves_seed`` (these have ``method`` set and the ``contraction_curve`` field
    populated; the caller can split them off if desired).
    """
    rng = np.random.default_rng(seed)
    glm = _build_glm(cfg)
    n_total = cfg.n_init + cfg.pool_size
    X_all = glm.sample_X(n_total, rng)
    w_star = glm.true_params(rng)
    Y_all = glm.sample_Y(X_all, w_star, rng)

    perm = rng.permutation(n_total)
    init_idx = perm[: cfg.n_init].tolist()
    pool_idx = perm[cfg.n_init :].tolist()

    init_theta = glm.fit(X_all[init_idx], Y_all[init_idx])

    method_states: Dict[str, Dict] = {}
    method_rngs: Dict[str, np.random.Generator] = {}
    for m in cfg.methods:
        method_states[m] = {
            "S": list(init_idx),
            "U": list(pool_idx),
            "theta": init_theta.copy(),
        }
        # independent RNG per method, deterministically derived from the seed
        method_rngs[m] = np.random.default_rng(np.random.SeedSequence((seed, hash(m) & 0xFFFFFFFF)))

    rows: List[Dict] = []
    decay_curves: List[Dict] = []

    for t in range(cfg.n_rounds):
        for m in cfg.methods:
            mrng = method_rngs[m]
            state = method_states[m]
            theta = state["theta"]
            U = np.array(state["U"], dtype=np.int64)
            if U.size == 0:
                break
            X_U = X_all[U]

            y_pseudo = _pseudo_targets(glm, theta, X_U, cfg.pseudo_jitter, mrng)
            g_clean = glm.batch_gradient(theta, X_U, y_pseudo)
            X_adv = fgsm_perturb(glm, theta, X_U, y_pseudo, cfg.eps_acq)
            g_adv = glm.batch_gradient(theta, X_adv, y_pseudo)

            local_idx = _select(
                m, g_clean=g_clean, g_adv=g_adv, alpha=cfg.alpha, q=cfg.q, rng=mrng
            )
            B = U[local_idx]

            # theory metrics on the selected batch
            g_clean_B = g_clean[local_idx]
            g_adv_B = g_adv[local_idx]
            phi_B = np.concatenate([g_clean_B, g_adv_B - g_clean_B], axis=1)

            G_phi = secant_gram(phi_B)
            G_c = clean_gram(g_clean_B)

            S_idx = np.array(state["S"], dtype=np.int64)
            X_union = np.concatenate([X_all[S_idx], X_all[B]], axis=0)
            Y_union = np.concatenate([Y_all[S_idx], Y_all[B]], axis=0)
            H_avg = glm.average_hessian(theta, X_union)

            ld_secant = logdet(G_phi, cfg.ridge)
            lmin_phi = lambda_min(G_phi, cfg.ridge)
            lmin_c = lambda_min(G_c, cfg.ridge)
            lmin_H = lambda_min(H_avg, cfg.ridge)
            min_pos_H = min_positive_eig(H_avg)
            cond_H = condition_number(H_avg, cfg.ridge)

            slope, losses_curve = contraction_probe(
                glm, theta, X_union, Y_union,
                K=cfg.contraction_K, lr=cfg.contraction_lr,
            )

            loss_before = float(losses_curve[0])
            loss_after = float(losses_curve[-1])
            test_metric = glm.loss(theta, X_all, Y_all)

            rows.append({
                "experiment": "GLM_theory",
                "glm_family": glm.name,
                "seed": seed,
                "round": t,
                "method": m,
                "batch_size": cfg.q,
                "alpha": cfg.alpha,
                "eps_acq": cfg.eps_acq,
                "logdet_secant": ld_secant,
                "lambda_min_secant_gram": lmin_phi,
                "lambda_min_clean_gram": lmin_c,
                "lambda_min_hessian": lmin_H,
                "min_positive_hessian_eig": min_pos_H,
                "condition_hessian": cond_H,
                "train_loss_before": loss_before,
                "train_loss_after": loss_after,
                "empirical_contraction_slope": slope,
                "clean_or_task_metric": test_metric,
            })

            if seed == cfg.decay_curves_seed and t == cfg.n_rounds // 2:
                decay_curves.append({
                    "glm_family": glm.name,
                    "seed": seed,
                    "round": t,
                    "method": m,
                    "losses": losses_curve.copy(),
                })

            # query labels & update state
            state["S"] = state["S"] + B.tolist()
            state["U"] = [int(i) for i in state["U"] if i not in set(B.tolist())]
            X_S = X_all[np.array(state["S"], dtype=np.int64)]
            Y_S = Y_all[np.array(state["S"], dtype=np.int64)]
            try:
                state["theta"] = glm.fit(X_S, Y_S, theta_init=state["theta"])
            except np.linalg.LinAlgError as e:
                log.warning("fit failed for %s round %d (%s); keeping previous theta", m, t, e)

    return rows, decay_curves


def run_all(cfg: ALConfig) -> tuple[List[Dict], List[Dict]]:
    all_rows: List[Dict] = []
    all_decay: List[Dict] = []
    for seed in cfg.seeds:
        log.info("running %s seed=%d", cfg.glm_family, seed)
        rows, decay = run_one_seed(cfg, seed)
        all_rows.extend(rows)
        all_decay.extend(decay)
    return all_rows, all_decay

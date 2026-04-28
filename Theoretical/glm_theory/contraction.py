"""Empirical loss-contraction probe for verifying the spectral contraction theorem."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from glm_theory.glms import BaseGLM


def contraction_probe(
    glm: BaseGLM,
    theta_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    K: int = 100,
    lr: float = 0.01,
) -> Tuple[float, np.ndarray]:
    """Run K full-batch GD steps from ``theta_init`` on the union (S_t ∪ B) loss.

    Returns
    -------
    slope : float
        Estimated empirical contraction rate c such that
        log(L_k - L_min + ε) ≈ a - c · k.
    losses : np.ndarray
        Length-(K+1) array of loss values.
    """
    theta = theta_init.copy()
    losses = np.empty(K + 1)
    losses[0] = glm.loss(theta, X, y)
    for k in range(K):
        G = glm.batch_gradient(theta, X, y).mean(axis=0)
        theta = theta - lr * G
        losses[k + 1] = glm.loss(theta, X, y)

    L_min = float(losses.min())
    eps = 1e-12
    diffs = losses - L_min + eps
    keep = diffs > 1e-8
    if keep.sum() < 5:
        return float("nan"), losses
    log_diffs = np.log(diffs[keep])
    ks = np.arange(len(losses))[keep]
    A = np.vstack([ks, np.ones_like(ks, dtype=float)]).T
    coef, *_ = np.linalg.lstsq(A, log_diffs, rcond=None)
    slope = float(coef[0])
    return float(-slope), losses

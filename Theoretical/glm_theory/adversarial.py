"""FGSM-style acquisition-time perturbations for GLMs."""

from __future__ import annotations

import numpy as np

from glm_theory.glms import (
    BaseGLM,
    ExponentialGLM,
    GammaGLM,
    GaussianGLM,
    PoissonGLM,
)

_REGRESSION_FAMILIES = (GaussianGLM, PoissonGLM, ExponentialGLM, GammaGLM)


def fgsm_perturb(
    glm: BaseGLM,
    theta: np.ndarray,
    X: np.ndarray,
    y_pseudo: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Return x_adv ≈ argmax_{||δ||_∞ ≤ ε} ℓ(θ; x+δ, y_pseudo) via one-step FGSM.

    For regression GLMs whose pseudo-target equals the current prediction (loss-grad-w.r.t.-x
    vanishes), fall back to the natural-parameter direction sign(θ).
    """
    X_adv = np.empty_like(X)
    is_regression = isinstance(glm, _REGRESSION_FAMILIES)
    for i, (x, y_i) in enumerate(zip(X, y_pseudo)):
        gx = glm.gradient_x(theta, x, y_i)
        if is_regression and not np.any(np.abs(gx) > 1e-12):
            gx = theta
        sign = np.sign(gx)
        sign[sign == 0] = 1.0
        X_adv[i] = x + eps * sign
    return X_adv

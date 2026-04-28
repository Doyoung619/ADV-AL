"""GLM family implementations with closed-form gradient and Hessian.

Notation
--------
x ∈ R^d, theta is the parameter vector (Cd-flattened W for softmax).
Per-sample loss is denoted ℓ; the AL training loss is the sample mean of ℓ.

Each GLM exposes:
    sample_X / sample_Y   — synthetic data generators
    loss(theta, X, y)     — mean loss
    gradient_theta        — per-sample gradient w.r.t. theta
    hessian_theta         — per-sample Hessian w.r.t. theta (no y dependence for GLMs)
    gradient_x            — per-sample loss gradient w.r.t. x (used for FGSM)
    batch_gradient        — vectorised gradient_theta over a batch
    average_hessian       — sample-mean Hessian over a batch
    fit                   — Newton-IRLS / closed-form parameter fit
    predict / pseudo_label
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


_POISSON_ETA_CLAMP = 5.0


def _outer(x: np.ndarray) -> np.ndarray:
    return np.outer(x, x)


class BaseGLM:
    name: str = "base"
    is_classification: bool = False

    def __init__(self, d: int, C: int = 1, sigma: float = 1.0,
                 ridge: float = 1e-4, w_scale: float = 1.0):
        self.d = d
        self.C = C
        self.sigma = sigma
        self.ridge = ridge
        self.w_scale = w_scale

    @property
    def param_dim(self) -> int:
        return self.d

    def init_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=self.param_dim) * 0.01

    def true_params(self, rng: np.random.Generator) -> np.ndarray:
        w = rng.normal(size=self.param_dim)
        w *= self.w_scale / (np.linalg.norm(w) + 1e-12)
        return w

    def sample_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        X = rng.normal(size=(n, self.d))
        # mild standardization (already roughly N(0,1)); keeps Poisson stable.
        return X

    def sample_Y(self, X: np.ndarray, w_star: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def gradient_theta(self, theta: np.ndarray, x: np.ndarray, y) -> np.ndarray:
        raise NotImplementedError

    def hessian_theta(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient_x(self, theta: np.ndarray, x: np.ndarray, y) -> np.ndarray:
        raise NotImplementedError

    def batch_gradient(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-sample gradient over a batch. Shape: (n, param_dim)."""
        out = np.empty((X.shape[0], self.param_dim))
        for i, (x, y_i) in enumerate(zip(X, y)):
            out[i] = self.gradient_theta(theta, x, y_i)
        return out

    def average_hessian(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        H = np.zeros((self.param_dim, self.param_dim))
        n = X.shape[0]
        for x in X:
            H += self.hessian_theta(theta, x)
        return H / max(n, 1)

    def predict(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pseudo_label(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, theta_init: Optional[np.ndarray] = None,
            n_steps: int = 50, tol: float = 1e-7) -> np.ndarray:
        """Newton-IRLS: theta_{k+1} = theta_k - (H+ridge*I)^{-1} G."""
        if theta_init is None:
            theta = np.zeros(self.param_dim)
        else:
            theta = theta_init.copy()
        I = np.eye(self.param_dim)
        for _ in range(n_steps):
            G = self.batch_gradient(theta, X, y).mean(axis=0)
            H = self.average_hessian(theta, X) + self.ridge * I
            try:
                step = np.linalg.solve(H, G)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, G, rcond=None)[0]
            theta = theta - step
            if np.linalg.norm(step) < tol:
                break
        return theta


# ---------------------------------------------------------------------------
# Gaussian / linear regression
#
# y = w_*^T x + N(0, σ^2)
# ℓ(θ; x, y) = 0.5/σ^2 (θ^T x - y)^2
# ∇_θ ℓ = (θ^T x - y)/σ^2 · x
# ∇_θ^2 ℓ = (1/σ^2) x x^T
# ---------------------------------------------------------------------------
class GaussianGLM(BaseGLM):
    name = "gaussian"
    is_classification = False

    def sample_Y(self, X, w_star, rng):
        return X @ w_star + rng.normal(size=X.shape[0]) * self.sigma

    def loss(self, theta, X, y):
        r = X @ theta - y
        return float(0.5 / self.sigma ** 2 * np.mean(r ** 2))

    def gradient_theta(self, theta, x, y):
        r = float(theta @ x - y)
        return (r / self.sigma ** 2) * x

    def hessian_theta(self, theta, x):
        return (1.0 / self.sigma ** 2) * _outer(x)

    def gradient_x(self, theta, x, y):
        r = float(theta @ x - y)
        return (r / self.sigma ** 2) * theta

    def batch_gradient(self, theta, X, y):
        r = X @ theta - y
        return (r / self.sigma ** 2)[:, None] * X

    def average_hessian(self, theta, X):
        n = max(X.shape[0], 1)
        return (X.T @ X) / (n * self.sigma ** 2)

    def predict(self, theta, X):
        return X @ theta

    def pseudo_label(self, theta, X):
        # mean prediction; AL loop adds frozen jitter (see al_loop) so g_c is non-zero.
        return X @ theta

    def fit(self, X, y, theta_init=None, n_steps=None, tol=None):
        d = self.d
        A = (X.T @ X) / self.sigma ** 2 + self.ridge * np.eye(d)
        b = (X.T @ y) / self.sigma ** 2
        return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Binary logistic regression
#
# p_θ(y=1|x) = σ(θ^T x), ℓ = -y log p - (1-y) log(1-p)
# ∇_θ ℓ = (p - y) x; ∇_θ^2 ℓ = p(1-p) x x^T
# ---------------------------------------------------------------------------
class LogisticGLM(BaseGLM):
    name = "logistic"
    is_classification = True

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def sample_Y(self, X, w_star, rng):
        p = self._sigmoid(X @ w_star)
        return (rng.uniform(size=X.shape[0]) < p).astype(np.float64)

    def loss(self, theta, X, y):
        z = X @ theta
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def gradient_theta(self, theta, x, y):
        p = float(self._sigmoid(theta @ x))
        return (p - float(y)) * x

    def hessian_theta(self, theta, x):
        p = float(self._sigmoid(theta @ x))
        return p * (1 - p) * _outer(x)

    def gradient_x(self, theta, x, y):
        p = float(self._sigmoid(theta @ x))
        return (p - float(y)) * theta

    def batch_gradient(self, theta, X, y):
        p = self._sigmoid(X @ theta)
        return (p - y)[:, None] * X

    def average_hessian(self, theta, X):
        p = self._sigmoid(X @ theta)
        w = p * (1 - p)  # (n,)
        # H = (1/n) X^T diag(w) X
        return (X.T * w) @ X / max(X.shape[0], 1)

    def predict(self, theta, X):
        return self._sigmoid(X @ theta)

    def pseudo_label(self, theta, X):
        return (self.predict(theta, X) >= 0.5).astype(np.float64)


# ---------------------------------------------------------------------------
# Multiclass softmax regression
#
# W ∈ R^{C×d}, θ = vec(W) row-major.
# p = softmax(Wx); ℓ = -log p_y
# ∇_W ℓ = (p - e_y) x^T  (so vec component [c*d+j] = (p_c - 1[c=y]) x_j)
# ∇_W^2 ℓ = (diag(p) - p p^T) ⊗ (x x^T)  (Cd × Cd)
# Hessian is rank-deficient (over-parameterized); always combined with ridge.
# ---------------------------------------------------------------------------
class SoftmaxGLM(BaseGLM):
    name = "softmax"
    is_classification = True

    @property
    def param_dim(self):
        return self.C * self.d

    def init_params(self, rng):
        return rng.normal(size=(self.C, self.d)).reshape(-1) * 0.01

    def true_params(self, rng):
        W = rng.normal(size=(self.C, self.d))
        W *= self.w_scale / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        return W.reshape(-1)

    def _W(self, theta):
        return theta.reshape(self.C, self.d)

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=-1, keepdims=True)
        e = np.exp(Z)
        return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-30)

    def sample_Y(self, X, w_star, rng):
        W = w_star.reshape(self.C, self.d)
        p = self._softmax(X @ W.T)
        cumul = np.cumsum(p, axis=1)
        u = rng.uniform(size=(X.shape[0], 1))
        return (u < cumul).argmax(axis=1).astype(np.int64)

    def loss(self, theta, X, y):
        W = self._W(theta)
        logits = X @ W.T
        m = logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(logits - m).sum(axis=1, keepdims=True)) + m
        return float(np.mean(lse.squeeze(1) - logits[np.arange(len(y)), y.astype(int)]))

    def gradient_theta(self, theta, x, y):
        W = self._W(theta)
        p = self._softmax((x @ W.T).reshape(1, -1)).reshape(-1)
        e = np.zeros(self.C)
        e[int(y)] = 1.0
        return np.outer(p - e, x).reshape(-1)

    def hessian_theta(self, theta, x):
        W = self._W(theta)
        p = self._softmax((x @ W.T).reshape(1, -1)).reshape(-1)
        A = np.diag(p) - np.outer(p, p)
        B = _outer(x)
        return np.kron(A, B)

    def gradient_x(self, theta, x, y):
        W = self._W(theta)
        p = self._softmax((x @ W.T).reshape(1, -1)).reshape(-1)
        e = np.zeros(self.C)
        e[int(y)] = 1.0
        return (p - e) @ W

    def batch_gradient(self, theta, X, y):
        W = self._W(theta)
        P = self._softmax(X @ W.T)
        E = np.zeros_like(P)
        E[np.arange(len(y)), y.astype(int)] = 1.0
        diff = P - E  # (n, C)
        # gradient[i, c*d+j] = diff[i, c] * X[i, j]
        return (diff[:, :, None] * X[:, None, :]).reshape(X.shape[0], -1)

    def average_hessian(self, theta, X):
        W = self._W(theta)
        P = self._softmax(X @ W.T)
        n = max(X.shape[0], 1)
        H = np.zeros((self.param_dim, self.param_dim))
        for i in range(X.shape[0]):
            A = np.diag(P[i]) - np.outer(P[i], P[i])
            B = _outer(X[i])
            H += np.kron(A, B)
        return H / n

    def predict(self, theta, X):
        return self._softmax(X @ self._W(theta).T)

    def pseudo_label(self, theta, X):
        return self.predict(theta, X).argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Poisson regression
#
# y ~ Poisson(exp(η)), η = w_*^T x.
# ℓ(θ; x, y) = exp(θ^T x) - y θ^T x
# ∇_θ ℓ = (exp(η) - y) x;  ∇_θ^2 ℓ = exp(η) x x^T
# Natural parameter is clamped to keep exp() stable.
# ---------------------------------------------------------------------------
class PoissonGLM(BaseGLM):
    name = "poisson"
    is_classification = False

    def sample_Y(self, X, w_star, rng):
        eta = np.clip(X @ w_star, -_POISSON_ETA_CLAMP, _POISSON_ETA_CLAMP)
        return rng.poisson(np.exp(eta)).astype(np.float64)

    def _eta(self, theta, X):
        return np.clip(X @ theta, -_POISSON_ETA_CLAMP, _POISSON_ETA_CLAMP)

    def loss(self, theta, X, y):
        eta = self._eta(theta, X)
        return float(np.mean(np.exp(eta) - y * eta))

    def gradient_theta(self, theta, x, y):
        eta = float(np.clip(theta @ x, -_POISSON_ETA_CLAMP, _POISSON_ETA_CLAMP))
        return (math.exp(eta) - float(y)) * x

    def hessian_theta(self, theta, x):
        eta = float(np.clip(theta @ x, -_POISSON_ETA_CLAMP, _POISSON_ETA_CLAMP))
        return math.exp(eta) * _outer(x)

    def gradient_x(self, theta, x, y):
        eta = float(np.clip(theta @ x, -_POISSON_ETA_CLAMP, _POISSON_ETA_CLAMP))
        return (math.exp(eta) - float(y)) * theta

    def batch_gradient(self, theta, X, y):
        eta = self._eta(theta, X)
        return (np.exp(eta) - y)[:, None] * X

    def average_hessian(self, theta, X):
        eta = self._eta(theta, X)
        w = np.exp(eta)
        return (X.T * w) @ X / max(X.shape[0], 1)

    def predict(self, theta, X):
        return np.exp(self._eta(theta, X))

    def pseudo_label(self, theta, X):
        return self.predict(theta, X)


GLM_REGISTRY = {
    "gaussian": GaussianGLM,
    "logistic": LogisticGLM,
    "softmax": SoftmaxGLM,
    "poisson": PoissonGLM,
}

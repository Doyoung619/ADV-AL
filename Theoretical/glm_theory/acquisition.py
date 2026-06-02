"""Acquisition methods for the GLM theory experiments.

All methods return indices into the unlabeled pool ``U``.

Conventions
-----------
- The ``alpha`` argument follows the Empirical-side ``p{alpha}`` convention: a candidate is
  retained if ``||g_clean|| >= percentile(||g_clean||, alpha)``. So ``alpha=75`` keeps the
  top 25% by clean-gradient norm. This matches the existing ``ours_badge_secant_p75`` runs.
- ``_kmeans_pp_seed`` performs k-means++ seeding only (no Lloyd refinement), matching the
  BADGE convention.
"""

from __future__ import annotations

import numpy as np


def _kmeans_pp_seed(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    norms = np.einsum("ij,ij->i", features, features)
    idx0 = int(rng.integers(n))
    selected = [idx0]
    d2 = norms + norms[idx0] - 2.0 * (features @ features[idx0])
    d2 = np.maximum(d2, 0.0)
    d2[idx0] = 0.0

    for _ in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            mask = np.ones(n, dtype=bool)
            mask[selected] = False
            remaining = np.where(mask)[0]
            if remaining.size == 0:
                break
            idx = int(rng.choice(remaining))
        else:
            probs = d2 / total
            idx = int(rng.choice(n, p=probs))
        selected.append(idx)
        new_d2 = norms + norms[idx] - 2.0 * (features @ features[idx])
        new_d2 = np.maximum(new_d2, 0.0)
        d2 = np.minimum(d2, new_d2)
    return np.array(selected, dtype=np.int64)


def random_acquire(rng: np.random.Generator, pool_size: int, q: int) -> np.ndarray:
    return rng.choice(pool_size, size=min(q, pool_size), replace=False)


def badge_acquire(g_clean: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    return _kmeans_pp_seed(g_clean, q, rng)


def secant_badge_acquire(
    g_clean: np.ndarray,
    g_adv: np.ndarray,
    alpha: float,
    q: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uncertainty-filtered secant BADGE.

    Steps (matching the paper's practical algorithm):
        1. Score by ||g_clean(x)||.
        2. Keep candidates with score >= alpha-th percentile.
        3. Lift kept points to φ(x) = [g_clean(x); Γ(x)] with Γ = g_adv - g_clean.
        4. Run k-means++ seeding in φ-space to pick q points.
    """
    norms = np.linalg.norm(g_clean, axis=1)
    threshold = float(np.percentile(norms, alpha))
    keep_mask = norms >= threshold
    if keep_mask.sum() < q:
        order = np.argsort(-norms)
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[order[: max(q, int(0.25 * len(norms)))]] = True
    keep_idx = np.where(keep_mask)[0]
    Gamma = g_adv[keep_idx] - g_clean[keep_idx]
    phi = np.concatenate([g_clean[keep_idx], Gamma], axis=1)
    chosen_local = _kmeans_pp_seed(phi, q, rng)
    return keep_idx[chosen_local]


def oracle_logdet_acquire(phi: np.ndarray, q: int, ridge: float = 1e-6) -> np.ndarray:
    """Greedy log-det maximization in φ-space (optional oracle baseline).

    Maximizes F_λ(B) = log det(λ I + Σ_{x∈B} φ(x) φ(x)^T) by greedy addition. Uses the
    matrix determinant lemma: argmax_i φ_i^T M^{-1} φ_i.
    """
    n, d = phi.shape
    M = ridge * np.eye(d)
    selected: list[int] = []
    sel_set: set[int] = set()
    for _ in range(q):
        Minv = np.linalg.inv(M)
        # gain_i = log(1 + φ_i^T M^{-1} φ_i); argmax = argmax of φ_i^T M^{-1} φ_i.
        scores = np.einsum("ij,jk,ik->i", phi, Minv, phi)
        scores[list(sel_set)] = -np.inf
        idx = int(np.argmax(scores))
        if not np.isfinite(scores[idx]):
            break
        selected.append(idx)
        sel_set.add(idx)
        ph = phi[idx]
        M = M + np.outer(ph, ph)
    return np.array(selected, dtype=np.int64)

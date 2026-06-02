"""Spectral metrics for the secant Gram, clean Gram, and Hessian matrices."""

from __future__ import annotations

import math

import numpy as np


def secant_gram(phi_batch: np.ndarray) -> np.ndarray:
    q = max(phi_batch.shape[0], 1)
    return (phi_batch.T @ phi_batch) / q


def clean_gram(g_clean_batch: np.ndarray) -> np.ndarray:
    q = max(g_clean_batch.shape[0], 1)
    return (g_clean_batch.T @ g_clean_batch) / q


def logdet(M: np.ndarray, ridge: float = 1e-6) -> float:
    A = M + ridge * np.eye(M.shape[0])
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0 or not math.isfinite(ld):
        return float("nan")
    return float(ld)


def lambda_min(M: np.ndarray, ridge: float = 1e-6) -> float:
    A = 0.5 * (M + M.T) + ridge * np.eye(M.shape[0])
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return float("nan")
    val = float(w.min())
    return val if math.isfinite(val) else float("nan")


def min_positive_eig(M: np.ndarray, tol: float = 1e-8) -> float:
    A = 0.5 * (M + M.T)
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return float("nan")
    pos = w[w > tol]
    if pos.size == 0:
        return float("nan")
    return float(pos.min())


def condition_number(M: np.ndarray, ridge: float = 1e-6) -> float:
    A = 0.5 * (M + M.T) + ridge * np.eye(M.shape[0])
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return float("nan")
    if w.min() <= 0:
        return float("nan")
    return float(w.max() / w.min())

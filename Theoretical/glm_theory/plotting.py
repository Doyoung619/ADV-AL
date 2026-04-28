"""Figures for Experiments G / H / I."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHOD_COLORS = {
    "random": "#888888",
    "badge": "#1f77b4",
    "secant_badge": "#d62728",
    "oracle_logdet": "#2ca02c",
}
METHOD_LABELS = {
    "random": "Random",
    "badge": "BADGE",
    "secant_badge": "Ours (Secant-BADGE)",
    "oracle_logdet": "Oracle-LogDet",
}
GLM_LABELS = {
    "gaussian": "Gaussian",
    "logistic": "Logistic",
    "softmax": "Softmax",
    "poisson": "Poisson",
}


def _hessian_eig_col(glm_family: str) -> str:
    # Softmax is rank-deficient; report the smallest positive eigenvalue.
    if glm_family == "softmax":
        return "min_positive_hessian_eig"
    return "lambda_min_hessian"


def _save(fig: plt.Figure, outpath: str) -> str:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_experiment_G(df: pd.DataFrame, glm_family: str, outdir: str) -> List[str]:
    """λ_min(G_B^φ) and λ_min(H_{S_t∪B}) over rounds (mean ± std over seeds)."""
    sub = df[df["glm_family"] == glm_family]
    if sub.empty:
        return []

    h_col = _hessian_eig_col(glm_family)
    metric_specs = [
        ("lambda_min_secant_gram", r"$\lambda_{\min}(G_B^{\phi})$"),
        (h_col, r"$\lambda_{\min}(H_{S_t \cup B})$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)
    for ax, (col, ylabel) in zip(axes, metric_specs):
        for method in sub["method"].unique():
            ms = sub[sub["method"] == method]
            grp = ms.groupby("round")[col].agg(["mean", "std"]).reset_index()
            ax.plot(
                grp["round"], grp["mean"],
                color=METHOD_COLORS.get(method, "k"),
                label=METHOD_LABELS.get(method, method),
                linewidth=1.8,
            )
            ax.fill_between(
                grp["round"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                color=METHOD_COLORS.get(method, "k"), alpha=0.18,
            )
        ax.set_xlabel("Active learning round")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_yscale("symlog", linthresh=1e-6)
    axes[0].legend(loc="best", frameon=True, framealpha=0.85)
    fig.suptitle(f"Experiment G — Spectral growth ({GLM_LABELS.get(glm_family, glm_family)})", y=1.02)

    outpath = os.path.join(outdir, f"experiment_G_spectral_growth_{glm_family}.pdf")
    return [_save(fig, outpath)]


def _scatter_with_corr(ax, x: np.ndarray, y: np.ndarray, methods: np.ndarray,
                       method_set: Sequence[str]):
    for m in method_set:
        mask = methods == m
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask],
            color=METHOD_COLORS.get(m, "k"), label=METHOD_LABELS.get(m, m),
            alpha=0.55, s=26, edgecolor="none",
        )
    finite = np.isfinite(x) & np.isfinite(y)
    pearson = spearman = float("nan")
    if finite.sum() >= 3:
        try:
            pearson = float(stats.pearsonr(x[finite], y[finite]).statistic)
        except Exception:
            pearson = float("nan")
        try:
            spearman = float(stats.spearmanr(x[finite], y[finite]).statistic)
        except Exception:
            spearman = float("nan")
        # trend line
        if np.isfinite(pearson):
            xs = np.linspace(x[finite].min(), x[finite].max(), 100)
            slope, intercept, *_ = stats.linregress(x[finite], y[finite])
            ax.plot(xs, slope * xs + intercept, color="black", linestyle="--",
                    linewidth=1.0, alpha=0.6)
    ax.grid(alpha=0.25)
    return pearson, spearman


def plot_experiment_H(df: pd.DataFrame, glm_family: str, outdir: str) -> List[str]:
    """Scatter F_λ(B) vs λ_min(G_B^φ), with Pearson/Spearman."""
    sub = df[df["glm_family"] == glm_family]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    methods = sub["method"].to_numpy()
    method_set = list(dict.fromkeys(methods.tolist()))
    p, s = _scatter_with_corr(
        ax, sub["logdet_secant"].to_numpy(), sub["lambda_min_secant_gram"].to_numpy(),
        methods, method_set,
    )
    ax.set_xlabel(r"$F_{\lambda}(B) = \log\det(\lambda I + G_B^{\phi})$")
    ax.set_ylabel(r"$\lambda_{\min}(G_B^{\phi})$")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_title(
        f"Experiment H — log-det vs $\\lambda_{{\\min}}$ "
        f"({GLM_LABELS.get(glm_family, glm_family)})\n"
        f"Pearson={p:.3f}  Spearman={s:.3f}"
    )
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    outpath = os.path.join(outdir, f"experiment_H_logdet_vs_lambdamin_{glm_family}.pdf")
    return [_save(fig, outpath)]


def plot_experiment_H_combined(df: pd.DataFrame, outdir: str) -> List[str]:
    families = [g for g in ["gaussian", "logistic", "softmax", "poisson"] if g in df["glm_family"].unique()]
    if not families:
        return []
    fig, axes = plt.subplots(1, len(families), figsize=(5.2 * len(families), 4.8), squeeze=False)
    for ax, fam in zip(axes[0], families):
        sub = df[df["glm_family"] == fam]
        methods = sub["method"].to_numpy()
        method_set = list(dict.fromkeys(methods.tolist()))
        p, s = _scatter_with_corr(
            ax, sub["logdet_secant"].to_numpy(), sub["lambda_min_secant_gram"].to_numpy(),
            methods, method_set,
        )
        ax.set_xlabel(r"$F_{\lambda}(B)$")
        ax.set_ylabel(r"$\lambda_{\min}(G_B^{\phi})$")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(f"{GLM_LABELS.get(fam, fam)}\nPearson={p:.3f} / Spearman={s:.3f}")
    axes[0][0].legend(loc="best", frameon=True, framealpha=0.85)
    fig.suptitle("Experiment H — log-det vs $\\lambda_{\\min}$ across GLMs", y=1.02)
    outpath = os.path.join(outdir, "experiment_H_logdet_vs_lambdamin_all_glms.pdf")
    return [_save(fig, outpath)]


def plot_experiment_I(df: pd.DataFrame, decay_curves: List[Dict], glm_family: str,
                      outdir: str) -> List[str]:
    """Scatter λ_min(H) vs empirical contraction slope; plus loss-decay panel for one seed."""
    sub = df[df["glm_family"] == glm_family]
    if sub.empty:
        return []
    h_col = _hessian_eig_col(glm_family)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    methods_arr = sub["method"].to_numpy()
    method_set = list(dict.fromkeys(methods_arr.tolist()))
    p, s = _scatter_with_corr(
        axes[0], sub[h_col].to_numpy(), sub["empirical_contraction_slope"].to_numpy(),
        methods_arr, method_set,
    )
    axes[0].set_xlabel(rf"$\lambda_{{\min}}(H_{{S_t \cup B}})$ ({h_col})")
    axes[0].set_ylabel("empirical contraction slope $c$")
    axes[0].set_xscale("symlog", linthresh=1e-6)
    axes[0].set_title(
        f"Experiment I — $\\lambda_{{\\min}}$ vs contraction "
        f"({GLM_LABELS.get(glm_family, glm_family)})\n"
        f"Pearson={p:.3f}  Spearman={s:.3f}"
    )
    axes[0].legend(loc="best", frameon=True, framealpha=0.85)

    fam_curves = [c for c in decay_curves if c["glm_family"] == glm_family]
    if fam_curves:
        for c in fam_curves:
            losses = np.asarray(c["losses"], dtype=float)
            L_min = float(losses.min())
            ld = np.log(np.maximum(losses - L_min + 1e-12, 1e-30))
            axes[1].plot(
                np.arange(len(losses)), ld,
                color=METHOD_COLORS.get(c["method"], "k"),
                label=METHOD_LABELS.get(c["method"], c["method"]),
                linewidth=1.8,
            )
        axes[1].set_xlabel("gradient step $k$")
        axes[1].set_ylabel(r"$\log(L_k - L_{\min} + \varepsilon)$")
        axes[1].set_title("Representative loss decay")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", frameon=True, framealpha=0.85)
    else:
        axes[1].set_axis_off()

    outpath = os.path.join(outdir, f"experiment_I_lambdamin_vs_contraction_{glm_family}.pdf")
    return [_save(fig, outpath)]


def plot_summary(df: pd.DataFrame, decay_curves: List[Dict], outdir: str) -> List[str]:
    families = [g for g in ["gaussian", "logistic", "softmax", "poisson"] if g in df["glm_family"].unique()]
    if not families:
        return []
    n = len(families)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 12.5), squeeze=False)

    for j, fam in enumerate(families):
        sub = df[df["glm_family"] == fam]
        h_col = _hessian_eig_col(fam)

        # Row 1: spectral growth λ_min(H)
        ax = axes[0][j]
        for method in sub["method"].unique():
            ms = sub[sub["method"] == method]
            grp = ms.groupby("round")[h_col].agg(["mean", "std"]).reset_index()
            ax.plot(grp["round"], grp["mean"],
                    color=METHOD_COLORS.get(method, "k"),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=1.6)
            ax.fill_between(grp["round"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                            color=METHOD_COLORS.get(method, "k"), alpha=0.18)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(alpha=0.25)
        ax.set_title(GLM_LABELS.get(fam, fam))
        if j == 0:
            ax.set_ylabel(r"$\lambda_{\min}(H)$")
            ax.legend(loc="best", fontsize=8, frameon=True)
        ax.set_xlabel("round")

        # Row 2: log-det vs λ_min scatter
        ax = axes[1][j]
        methods_arr = sub["method"].to_numpy()
        method_set = list(dict.fromkeys(methods_arr.tolist()))
        p, s = _scatter_with_corr(
            ax, sub["logdet_secant"].to_numpy(), sub["lambda_min_secant_gram"].to_numpy(),
            methods_arr, method_set,
        )
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(f"Pearson={p:.2f} / Spearman={s:.2f}", fontsize=10)
        if j == 0:
            ax.set_ylabel(r"$\lambda_{\min}(G_B^{\phi})$")
        ax.set_xlabel(r"$F_{\lambda}(B)$")

        # Row 3: λ_min(H) vs contraction slope
        ax = axes[2][j]
        p, s = _scatter_with_corr(
            ax, sub[h_col].to_numpy(), sub["empirical_contraction_slope"].to_numpy(),
            methods_arr, method_set,
        )
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_title(f"Pearson={p:.2f} / Spearman={s:.2f}", fontsize=10)
        if j == 0:
            ax.set_ylabel("contraction slope $c$")
        ax.set_xlabel(r"$\lambda_{\min}(H)$")

    fig.suptitle("GLM theory verification summary", y=1.005, fontsize=14)
    outpath = os.path.join(outdir, "glm_theory_summary.pdf")
    return [_save(fig, outpath)]


def correlation_summary(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    for fam, sub in df.groupby("glm_family"):
        for method, sub2 in sub.groupby("method"):
            x = sub2[x_col].to_numpy()
            y = sub2[y_col].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            n_ok = int(mask.sum())
            if n_ok < 3:
                p = s = float("nan")
            else:
                try:
                    p = float(stats.pearsonr(x[mask], y[mask]).statistic)
                except Exception:
                    p = float("nan")
                try:
                    s = float(stats.spearmanr(x[mask], y[mask]).statistic)
                except Exception:
                    s = float("nan")
            rows.append({
                "glm_family": fam,
                "method": method,
                "n": n_ok,
                "pearson": p,
                "spearman": s,
            })
    return pd.DataFrame(rows)


def experiment_G_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, sub in df.groupby("glm_family"):
        h_col = _hessian_eig_col(fam)
        for method, sub2 in sub.groupby("method"):
            final = sub2[sub2["round"] == sub2["round"].max()][h_col]
            rows.append({
                "glm_family": fam,
                "method": method,
                "final_lambda_min_H_mean": float(final.mean()),
                "final_lambda_min_H_std": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
                "all_round_lambda_min_H_mean": float(sub2[h_col].mean()),
                "all_round_lambda_min_H_std": float(sub2[h_col].std(ddof=1))
                    if len(sub2[h_col]) > 1 else 0.0,
            })
    return pd.DataFrame(rows)

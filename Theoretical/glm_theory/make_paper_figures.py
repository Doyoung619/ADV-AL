"""Generate paper-style figures for Experiments G and H (= renamed Experiment I).

Layout per the paper-prep spec:
  - GLM families: gaussian, softmax, poisson, exponential, gamma  (logistic excluded)
  - Methods:      Random, BADGE, Ours (Secant-BADGE) — exactly these three in legend
  - Title:        "GLM - <Family>"
  - Larger axis / title fonts
  - Experiment G:
      single panel: round (0..20) vs λ_min(H)
      softmax y-floor = 1e-3, log-y; non-softmax linear
  - Experiment H (renamed from I):
      single panel: gradient step (0..80) vs log(L_k - L_min + ε)

Cleanup:
  - Delete experiments/experimentH (old logdet vs λ_min content)
  - Rename experiments/experimentI → experiments/experimentH
  - Clear *.pdf in both figures/ folders, write the new set in their place
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
GLM_FAMILIES = ["gaussian", "logistic", "softmax", "poisson", "exponential", "gamma"]

GLM_DISPLAY = {
    "gaussian":    "GLM - Gaussian",
    "logistic":    "GLM - Binary",
    "softmax":     "GLM - Softmax",
    "poisson":     "GLM - Poisson",
    "exponential": "GLM - Exponential",
    "gamma":       "GLM - Gamma",
}

METHOD_ORDER = ["random", "badge", "secant_badge"]
METHOD_LABEL = {
    "random":       "Random",
    "badge":        "BADGE",
    "secant_badge": "Ours",
}
METHOD_COLOR = {
    "random":       "#888888",
    "badge":        "#1f77b4",
    "secant_badge": "#d62728",
}

PAPER_RC = {
    "axes.titlesize":   24,
    "axes.labelsize":   22,
    "xtick.labelsize":  18,
    "ytick.labelsize":  18,
    "legend.fontsize":  18,
}


# ---------------------------------------------------------------------------
def _load_metrics(staging_dirs: List[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for d in staging_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "metrics.csv"))):
            try:
                dfs.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                continue
    if not dfs:
        raise FileNotFoundError(f"no metrics.csv files in {staging_dirs}")
    df = pd.concat(dfs, ignore_index=True)
    return df


def _load_decay_curves(staging_dirs: List[str]) -> List[Dict]:
    out: List[Dict] = []
    for d in staging_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "decay_curves.pkl"))):
            try:
                with open(p, "rb") as f:
                    out.extend(pickle.load(f))
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
def plot_experiment_G(df: pd.DataFrame, glm_family: str, outdir: str) -> str:
    sub = df[(df["glm_family"] == glm_family)
             & (df["round"] >= 0) & (df["round"] <= 15)
             & (df["method"].isin(METHOD_ORDER))]
    if sub.empty:
        return ""
    h_col = "min_positive_hessian_eig" if glm_family == "softmax" else "lambda_min_hessian"

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        for method in METHOD_ORDER:
            ms = sub[sub["method"] == method]
            if ms.empty:
                continue
            grp = ms.groupby("round")[h_col].agg(["mean", "std"]).reset_index()
            ax.plot(grp["round"], grp["mean"],
                    color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method],
                    linewidth=2.4)
            ax.fill_between(grp["round"],
                            grp["mean"] - grp["std"],
                            grp["mean"] + grp["std"],
                            color=METHOD_COLOR[method], alpha=0.18)
        ax.set_xlim(0, 15)
        ax.set_xticks([0, 5, 10, 15])
        if glm_family == "softmax":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2)
        ax.set_xlabel("Active learning round")
        ax.set_ylabel(r"$\lambda_{\min}(H_{S_t \cup B})$")
        ax.set_title(GLM_DISPLAY[glm_family])
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        fig.tight_layout()
    out_path = os.path.join(outdir, f"experiment_G_{glm_family}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_experiment_H(decay_curves: List[Dict], glm_family: str, outdir: str) -> str:
    fam_curves = [c for c in decay_curves
                  if c.get("glm_family") == glm_family
                  and c.get("method") in METHOD_ORDER]
    if not fam_curves:
        return ""

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        for method in METHOD_ORDER:
            curves = [c for c in fam_curves if c["method"] == method]
            if not curves:
                continue
            losses_full = np.asarray(curves[0]["losses"], dtype=float)
            # Reference L_min from the FULL contraction probe (K=100), so truncating the
            # display window doesn't cause fast-converging methods to crash to log(ε) at
            # the endpoint.
            L_min = float(losses_full.min())
            losses = losses_full[:81]
            ld = np.log(np.maximum(losses - L_min + 1e-12, 1e-30))
            ax.plot(np.arange(len(losses)), ld,
                    color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method],
                    linewidth=2.4)
        ax.set_xlim(0, 80)
        ax.set_xlabel("Gradient step $k$")
        ax.set_ylabel(r"$\log(L_k - L_{\min} + \varepsilon)$")
        ax.set_title(GLM_DISPLAY[glm_family])
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
        fig.tight_layout()
    out_path = os.path.join(outdir, f"experiment_H_{glm_family}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def cleanup_dirs(root: str, dry_run: bool = False) -> None:
    """Delete old experimentH (logdet content), rename experimentI → experimentH,
    clear *.pdf inside both experimentG/figures/ and experimentH/figures/."""
    old_H = os.path.join(root, "experimentH")
    old_I = os.path.join(root, "experimentI")

    if os.path.exists(old_H):
        print(f"  rm -rf {old_H}")
        if not dry_run:
            shutil.rmtree(old_H)

    if os.path.exists(old_I):
        print(f"  mv {old_I} → {old_H}")
        if not dry_run:
            os.rename(old_I, old_H)

    for fig_dir in (os.path.join(root, "experimentG", "figures"),
                    os.path.join(old_H, "figures")):
        if os.path.exists(fig_dir):
            pdfs = [f for f in os.listdir(fig_dir) if f.endswith(".pdf")]
            if pdfs:
                print(f"  clear {len(pdfs)} pdfs in {fig_dir}")
                if not dry_run:
                    for f in pdfs:
                        os.remove(os.path.join(fig_dir, f))
        else:
            if not dry_run:
                os.makedirs(fig_dir, exist_ok=True)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dirs", nargs="+", default=[
        "experiments/_glm_theory_runs",
        "experiments/_glm_theory_ext_R20",
    ], help="Staging dirs to load metrics + decay curves from. R=20 only.")
    parser.add_argument("--root", default="experiments",
                        help="Root containing experimentG / experimentH / experimentI.")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip the rename + clear step (just regenerate figures).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = _load_metrics(args.staging_dirs)
    decay = _load_decay_curves(args.staging_dirs)
    print(f"loaded {len(df)} metric rows / {len(decay)} decay curves "
          f"from {len(args.staging_dirs)} staging dirs")
    print(f"families: {sorted(df.glm_family.unique().tolist())}")

    if not args.no_cleanup:
        print("[cleanup]")
        cleanup_dirs(args.root, dry_run=args.dry_run)

    G_dir = os.path.join(args.root, "experimentG", "figures")
    H_dir = os.path.join(args.root, "experimentH", "figures")
    if not args.dry_run:
        os.makedirs(G_dir, exist_ok=True)
        os.makedirs(H_dir, exist_ok=True)

    print("[figures]")
    for glm in GLM_FAMILIES:
        if args.dry_run:
            print(f"  would write G/{glm} and H/{glm}")
            continue
        p = plot_experiment_G(df, glm, G_dir)
        if p:
            print(f"  wrote {p}")
        else:
            print(f"  WARN: no G data for {glm}")
        p = plot_experiment_H(decay, glm, H_dir)
        if p:
            print(f"  wrote {p}")
        else:
            print(f"  WARN: no decay curves for {glm}")


if __name__ == "__main__":
    main()

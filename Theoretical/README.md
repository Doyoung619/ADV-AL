# Theoretical / GLM theory verification

This directory contains the controlled synthetic GLM experiment suite that verifies the
theory section of *"Contraction Based Uncertainty Aware Learning for Robust Model"*. It is
deliberately separate from `Empirical/`: the goal here is **not** benchmark accuracy but
verifying three theoretical claims about the practical Secant-BADGE acquisition algorithm.

## Layout

```
Theoretical/
├── README.md
└── glm_theory/
    ├── glms.py                 # Gaussian / Logistic / Softmax / Poisson GLM primitives
    ├── adversarial.py          # FGSM-style acquisition-time perturbation
    ├── acquisition.py          # Random / BADGE / Secant-BADGE / Oracle-LogDet
    ├── theory_metrics.py       # log-det, λ_min, condition number
    ├── contraction.py          # full-batch GD probe for empirical contraction rate
    ├── al_loop.py              # AL driver maintaining per-method state
    ├── plotting.py             # figures for Experiments G / H / I
    └── run_glm_theory_experiments.py    # CLI entry-point
```

## Method (Secant-BADGE — exactly as in the paper)

For each unlabeled `x`:

1. Pseudo-label `ŷ = argmax_y p_θ(y|x)` (or `μ_θ(x)` for regression; regression GLMs
   add a small frozen jitter to keep the clean gradient non-degenerate).
2. Compute clean GLM gradient `g_c(x) = ∇_θ ℓ(θ; x, ŷ)`.
3. Keep the **top-α-percentile** candidates by `||g_c(x)||`. The convention follows the
   `Empirical/` side: `alpha=75` ⇒ keep the top 25%.
4. For retained `x`, compute `δ*_t(x)` via one-step FGSM with budget `eps_acq`.
   Regression GLMs whose loss-gradient w.r.t. `x` vanishes fall back to
   `δ = eps_acq · sign(θ)` (natural-parameter direction).
5. Compute adversarial gradient `g_a(x) = ∇_θ ℓ(θ; x_adv, ŷ)`.
6. Form `Γ(x) = g_a(x) - g_c(x)` and the lifted secant embedding
   `φ(x) = [g_c(x); Γ(x)]`.
7. Select the batch of size `q` via **k-means++ seeding** in φ-space (no Lloyd
   refinement). The optional `oracle_logdet` baseline replaces step 7 with greedy
   log-det maximisation; it is reported only as an oracle and is **not** the default.

## Experiments

| ID | Title | What it verifies |
|----|-------|------------------|
| G  | Spectral Growth | Selected batches grow `λ_min(G_B^φ)` and `λ_min(H_{S_t ∪ B})` round-over-round. |
| H  | Log-det vs λ_min | `F_λ(B) = log det(λI + G_B^φ)` is positively correlated with `λ_min(G_B^φ)`. |
| I  | λ_min vs Contraction | `λ_min(H_{S_t ∪ B})` predicts the empirical contraction rate. |

## Running

```bash
python Theoretical/glm_theory/run_glm_theory_experiments.py \
    --glm gaussian logistic softmax poisson \
    --seeds 0 1 2 3 4 \
    --rounds 20 \
    --batch-size 50 \
    --pool-size 2000 \
    --init-size 100 \
    --dim 20 \
    --alpha 75 \
    --eps-acq 0.25 \
    --outdir results/glm_theory
```

CPU is fine; dimensions are small (`d=20`, `C=5`, `q=50`, pool 2000). Pass
`--methods random badge secant_badge oracle_logdet` to include the Oracle-LogDet
baseline.

## Outputs

```
results/glm_theory/
├── config.json
├── metrics.csv                    # one row per (glm, seed, round, method)
├── decay_curves.pkl               # per-method loss curves at the mid-round of seed 0
├── summary_experiment_G.csv
├── summary_experiment_H_correlations.csv
├── summary_experiment_I_correlations_lambdaH.csv
├── summary_experiment_I_correlations_minposH.csv
└── figures/
    ├── experiment_G_spectral_growth_<glm>.pdf
    ├── experiment_H_logdet_vs_lambdamin_<glm>.pdf
    ├── experiment_H_logdet_vs_lambdamin_all_glms.pdf
    ├── experiment_I_lambdamin_vs_contraction_<glm>.pdf
    └── glm_theory_summary.pdf
```

For softmax (whose Hessian is rank-deficient by construction), Experiment-G and
Experiment-I plots use `min_positive_hessian_eig` as the spectral floor; for the other
three GLMs the ridge-stabilised `lambda_min_hessian` is used.

## Key columns in `metrics.csv`

`experiment, glm_family, seed, round, method, batch_size, alpha, eps_acq,
logdet_secant, lambda_min_secant_gram, lambda_min_clean_gram, lambda_min_hessian,
min_positive_hessian_eig, condition_hessian, train_loss_before, train_loss_after,
empirical_contraction_slope, clean_or_task_metric`

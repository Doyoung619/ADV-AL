# CIFAR-10 Active Learning: BADGE + Dual Constraints

This codebase runs a shared active learning pipeline on CIFAR-10 with a small CNN and compares:

1. `badge` (existing baseline, unchanged)
2. `badge_dual_a` (robustness-constrained BADGE)
3. `badge_dual_b` (BADGE-constrained robustness)

## Default Experiment Settings

- Dataset: `cifar10`
- Model: `small_cnn`
- Initial labeled size: `200`
- Acquisition batch size: `50`
- Rounds: `20`
- Epochs per round: `50`
- Retrain from scratch each round: enabled
- Acquisition epsilon: `epsilon_acq = 1/255`
- Acquisition attack type: `fgsm` (optional `pgd`)
- Robust eval epsilon: `8/255`
- GPU: used automatically when available
- Seeded execution with deterministic mode enabled by default

## Robustness Score Definition (`c(x)`)

Implemented as:

`c(x) = max_{||delta||_inf <= epsilon_acq} || z(x + delta) - z(x) ||_2^2`

where `z(.)` are logits (pre-softmax).

Implementation details:
- clean branch is detached: `z_clean = model(x).detach()`
- only `delta` is optimized (model weights are not updated)
- objective is logit mismatch only (no CE, no labels)
- FGSM: one ascent step on `delta`
- PGD: random init in `L_inf` ball + projected ascent steps
- `x + delta` is clamped to valid input range
- returns one scalar `c(x)` per sample

## Methods

### `badge`
- Uses existing BADGE gradient embeddings
- Uses BADGE k-means++ selection

### `badge_dual_a`
- Compute `c(x)` for all unlabeled samples
- Set `kappa_c = quantile(c, 0.5)`
- Feasible set: `F_c = {x | c(x) >= kappa_c}`
- Run standard BADGE k-means++ only on `F_c`

### `badge_dual_b`
- Compute BADGE embedding norm `b(x) = ||g_BADGE(x)||_2`
- Set `kappa_b = quantile(b, 0.5)`
- Feasible set: `F_b = {x | b(x) >= kappa_b}`
- Compute `c(x)` and select top-`B` by `c(x)` within `F_b`

### `logdet_adv_disp`
- Computes adversarial semantic displacement vectors:
  - `Delta(x) = z(x + delta*(x)) - z(x)`
  - `delta*(x)` approximately maximizes `||z(x+delta)-z(x)||_2^2` under `||delta||_inf <= epsilon`
- Embedding for this method is logits (`g(x) = z(x)`).
- Selects a batch greedily by maximizing:
  - `log det(lambda I + sum_{x in B} Delta(x) Delta(x)^T)`
- Greedy marginal score at each step:
  - `s(x) = Delta(x)^T A^{-1} Delta(x)`
  - with rank-1 Sherman-Morrison inverse update.
- Default method hyperparameters:
  - `--logdet-adv-disp-attack fgsm`
  - `--logdet-adv-disp-epsilon 1/255`
  - `--logdet-adv-disp-lambda 1e-3`
  - `--logdet-adv-disp-pgd-steps 5`
  - `--logdet-adv-disp-pgd-step-size None` (auto = `epsilon / max(steps/2, 1)`)
  - `--logdet-adv-disp-pgd-random-start`

## Logging and Outputs

Per round, logs include:
- method name, round, labeled size
- `c(x)` stats: min/max/mean/std
- `b(x)` stats: min/max/mean/std
- quantile threshold(s) and feasible-set size
- selected sample indices
- timing breakdown (scoring/selection/training/eval/round total)

Per-round CSVs are saved under `selected_indices/` with columns:
- `sample_index`
- `c_score`
- `b_score`
- `feasible_flag`
- `selected_flag`

Evaluation per round includes:
- clean accuracy
- FGSM robust accuracy (`epsilon=8/255`)
- PGD robust accuracy (default 10 steps)
- average logit mismatch
- acquisition/training/eval/round timing

## Commands

Single run:

```bash
python main.py --acquisition_method badge
python main.py --acquisition_method badge_dual_a
python main.py --acquisition_method badge_dual_b
python main.py --acquisition_method logdet_adv_disp \
  --logdet-adv-disp-attack fgsm \
  --logdet-adv-disp-epsilon 0.0039215686 \
  --logdet-adv-disp-lambda 1e-3
python main.py --acquisition_method logdet_adv_disp \
  --logdet-adv-disp-attack pgd \
  --logdet-adv-disp-epsilon 0.0039215686 \
  --logdet-adv-disp-pgd-steps 10 \
  --logdet-adv-disp-pgd-step-size 0.0007843137 \
  --logdet-adv-disp-pgd-random-start
```

Full sweep:

```bash
for method in badge badge_dual_a badge_dual_b; do
  for seed in 0 1 2; do
    python main.py \
      --dataset cifar10 \
      --model small_cnn \
      --acquisition_method ${method} \
      --initial_labeled_size 200 \
      --acquisition_batch_size 50 \
      --rounds 20 \
      --epochs_per_round 50 \
      --epsilon_acq 0.0039215686 \
      --seed ${seed}
  done
done
```

Windows PowerShell equivalent:

```powershell
foreach ($method in @("badge", "badge_dual_a", "badge_dual_b")) {
  foreach ($seed in @(0, 1, 2)) {
    python main.py `
      --dataset cifar10 `
      --model small_cnn `
      --acquisition_method $method `
      --initial_labeled_size 200 `
      --acquisition_batch_size 50 `
      --rounds 20 `
      --epochs_per_round 50 `
      --epsilon_acq 0.0039215686 `
      --seed $seed
  }
}
```

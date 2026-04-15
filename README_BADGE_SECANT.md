# Robust BADGE Secant Notes

This branch adds two robust BADGE acquisition variants:

- `ours_badge_secant`
- `ours_badge_jointadv`

The original `ours_secant_badge` method is still available. The new
`ours_badge_secant` name uses the same secant embedding but matches the
experiment naming used in recent runs.

## Embeddings

Let `g_clean(x)` be the analytic last-layer BADGE gradient embedding on the
clean input, using the model prediction as the pseudo-label. Let `g_adv(x)` be
the same embedding after an acquisition-time adversarial attack, with the clean
pseudo-label held fixed.

`ours_badge_secant` uses:

```text
psi_secant(x) = [g_clean(x); g_adv(x) - g_clean(x)]
```

`ours_badge_jointadv` uses:

```text
psi_jointadv(x) = [g_clean(x); g_adv(x)]
```

Both methods then run the existing BADGE k-means++ selector on the robust
embedding.

## Prefiltering

`ours_badge_secant` supports a clean-anchor secant filter:

```bash
--prefilter-metric secant_clean_grad_norm
--prefilter-drop-percent 75
```

This drops the bottom percentile by `||g_clean(x)||_2` before BADGE selection,
while always retaining at least the acquisition budget.

`ours_badge_jointadv` supports:

```bash
--prefilter-metric joint_embedding_norm
```

If `--prefilter-drop-percent` is positive and no explicit prefilter metric is
provided, the parser chooses the default metric for the robust BADGE method.

## Single Run Examples

Base secant BADGE:

```bash
python main.py \
  --dataset cifar10 \
  --model resnet18 \
  --acquisition-method ours_badge_secant \
  --prefilter-metric none \
  --prefilter-drop-percent 0 \
  --initial-labeled-size 500 \
  --acquisition-size 100 \
  --num-rounds 20 \
  --epochs-per-round 80 \
  --epsilon 0.00392156862745098 \
  --seed 0 \
  --output-dir experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_ours_badge_secant_b100_r20 \
  --run-name seed_0 \
  --train-mode adv \
  --adv-train-attack pgd \
  --adv-train-epsilon 0.00392156862745098 \
  --adv-train-steps 3 \
  --data-dir data \
  --num-workers 8 \
  --pin-memory \
  --skip-logit-mismatch-eval \
  --no-save-checkpoints
```

p75 clean-anchor prefilter:

```bash
python main.py \
  --dataset cifar10 \
  --model resnet18 \
  --acquisition-method ours_badge_secant \
  --prefilter-metric secant_clean_grad_norm \
  --prefilter-drop-percent 75 \
  --initial-labeled-size 500 \
  --acquisition-size 100 \
  --num-rounds 20 \
  --epochs-per-round 80 \
  --epsilon 0.00392156862745098 \
  --seed 0 \
  --output-dir experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_ours_badge_secant_p75_b100_r20 \
  --run-name seed_0 \
  --train-mode adv \
  --adv-train-attack pgd \
  --adv-train-epsilon 0.00392156862745098 \
  --adv-train-steps 3 \
  --data-dir data \
  --num-workers 8 \
  --pin-memory \
  --skip-logit-mismatch-eval \
  --no-save-checkpoints
```

## Seed 5-9 Batch Plan

The seed 5-9 follow-up experiments can be generated from the same pattern
without committing large `experiments/` artifacts. The intended scope is:

- `main_resnet18_cifar10`: `badge`, `coreset`, `ours_badge_secant`, `ours_badge_secant` p75, seeds `5..9`
- `main_smallcnn_cifar10`: `badge`, `coreset`, `ours_badge_secant`, `ours_badge_secant` p75, seeds `5..9`
- `main_smallcnn_cifar100`: `badge`, `coreset`, `ours_badge_secant`, `ours_badge_secant` p75, seeds `5..9`
- `main_resnet18_cifar100`: `coreset` only, seeds `5..9`

Use a Slurm array with one GPU per task and `--array=0-7` to cap allocation at
eight GPUs. Keep runtime state, Slurm output, and generated experiment records
out of git unless a small manifest is explicitly needed.

## Validation

Run the smoke test in the project environment:

```bash
python smoke_test_ours_badge_robust_embeddings.py
```


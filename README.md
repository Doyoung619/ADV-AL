## ADV-AL — Active Learning for Adversarial Robustness

This repository contains the code, experiment harness, and result artifacts for
**Secant-BADGE** active learning under adversarial training. The selection rule
augments BADGE's gradient embedding with an *adversarial secant* term so that
each acquired batch increases not just gradient diversity but also the
spectral conditioning that governs robust learning.

The repository is organized into two complementary halves:

- [Empirical/](Empirical/) — full benchmark pipeline (CIFAR-10 / CIFAR-100 /
  TinyImageNet / SVHN / FashionMNIST × small_cnn / ResNet-18 / VGG-16) with
  PGD adversarial training and PGD/FGSM evaluation.
- [Theoretical/](Theoretical/) — controlled GLM (Gaussian / Logistic / Softmax /
  Poisson / Gamma / Exponential) verification suite for the three theoretical
  claims behind the method.

Per-component documentation already lives in
[Empirical/README.md](Empirical/README.md) and
[Theoretical/README.md](Theoretical/README.md). This top-level README focuses
on the **experiment grid** (`experimentMain` and `experimentA`–`experimentH`)
and how to reproduce it.

---

## What the method does

For each unlabeled `x`:

1. Pseudo-label `ŷ = argmax_y p_θ(y | x)`.
2. Compute the clean per-sample gradient `g_c(x) = ∇_θ ℓ(θ; x, ŷ)`.
3. Keep the **top-α-percentile** of candidates by `||g_c(x)||`
   (`alpha=75` ⇒ keep top 25%; `p10`/`p25`/`p50`/`p75`/`p90` variants).
4. For survivors, generate one-step FGSM (or multi-step PGD) perturbation
   `δ*(x)` under `||δ||_∞ ≤ ε_acq`.
5. Compute the adversarial gradient `g_a(x) = ∇_θ ℓ(θ; x + δ*, ŷ)`.
6. Form the secant `Γ(x) = g_a(x) − g_c(x)` and the lifted embedding
   `φ(x) = [g_c(x); Γ(x)]`.
7. Select a batch of size `q` via **k-means++ seeding** in φ-space.

The `Empirical/` side wraps this acquisition inside a PGD-trained active
learning loop; the `Theoretical/` side replaces the deep network with a closed-
form GLM so that `λ_min(H)`, `log det(λI + Gᴮ)`, and the empirical contraction
rate can all be measured exactly.

---

## Repository layout

```
ADV-AL/
├── Empirical/                      # benchmark pipeline (training + acquisition)
│   ├── main.py                     # single-run entry point
│   ├── config.py / configs/        # CLI args + experiment_sets/ JSONs
│   ├── runners/                    # active_learning_runner.py
│   ├── acquisition/                # one file per method (badge, ours_*, saal, …)
│   ├── attacks.py / train.py / eval.py
│   ├── models.py / datasets.py
│   └── scripts/
│       ├── build_experiment_sets.py        # expand a set JSON into commands
│       ├── launch_experiment_set.py        # run those commands across GPUs
│       ├── launch_all_experiments.py       # build + launch every set
│       ├── aggregate_experiment_set.py     # build per-set summary CSVs
│       ├── build_experimentA_E_graphs.py   # ablation figures
│       └── build_experimentF_bar_figures.py
├── Theoretical/                    # GLM verification suite
│   ├── glm_theory/                 # GLM primitives, AL loop, plotting
│   └── scripts/run_glm_theory_*.sh # SLURM launchers
└── experiments/                    # all completed runs and aggregated results
    ├── experimentMain/             # main empirical comparison (ADV training)
    ├── experimentA/ … experimentF/ # empirical ablations
    ├── experimentG/, experimentH/  # GLM theory verification outputs
    └── figures/{experimentA, …}/   # paper-ready figures aggregated per set
```

`experiments/` is the read-only artifact store. `Empirical/` and
`Theoretical/` are what you edit and run; their outputs land back in
`experiments/<set_name>/…`.

---

## The experiment grid

Each entry below is a self-contained sweep written to its own subdirectory of
`experiments/`. Methods compared throughout: `random`, `entropy`, `BADGE`,
`SAAL`, `ours_badge_secant` (Ours), and `ours_badge_secant_p75` (Ours+p75).
Empirical results come from PGD-adversarial training with `ε_train = 1/255`
and PGD-10 evaluation at `ε_eval = 8/255` unless otherwise noted.

| Set            | Question                                              | Sweep axis                                                        | Datasets                          | Models                          | Output dir                                |
| -------------- | ----------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------- | ------------------------------- | ----------------------------------------- |
| **Main**       | Headline: does Secant-BADGE win on robust acc?        | none — flagship comparison                                        | CIFAR-10, CIFAR-100, TinyImageNet | small_cnn, ResNet-18, VGG-16    | `experiments/experimentMain/`             |
| **A**          | How does the budget split affect ranking?             | acquisition batch × rounds with fixed total budget (50/40, 100/20, 200/10, 400/5) | CIFAR-10, CIFAR-100   | small_cnn                       | `experiments/experimentA/`                |
| **B**          | Does the initial labeled size change the ranking?     | `init ∈ {250, 500, 1000}` at acq=100, rounds=20                   | CIFAR-10, CIFAR-100               | small_cnn                       | `experiments/experimentB/`                |
| **C**          | How does the round horizon affect AUC vs final acc?   | `rounds ∈ {5, 10, 20, 30}` at init=500, acq=100                   | CIFAR-10, CIFAR-100               | small_cnn                       | `experiments/experimentC/`                |
| **D**          | Sensitivity to acquisition-time attack strength.      | `ε_acq ∈ {1, 2, 4, 8}/255`                                        | CIFAR-10, CIFAR-100               | small_cnn                       | `experiments/experimentD/`                |
| **E**          | Tight low-ε regime control for D.                     | `ε_acq = 1/255` reference replicate                               | CIFAR-10, CIFAR-100               | small_cnn                       | `experiments/experimentE/`                |
| **F**          | Percentile gate ablation for the secant filter.       | `p ∈ {10, 25, 50, 75, 90}`                                        | CIFAR-10, CIFAR-100               | small_cnn, ResNet-18            | `experiments/experimentF/`                |
| **G** (theory) | Does the selected batch grow `λ_min(G_ᴮ^φ)`?           | rounds × seed, six GLM families                                   | synthetic GLM                     | —                               | `experiments/experimentG/`                |
| **H** (theory) | Is `log det(λI + G_ᴮ^φ)` correlated with `λ_min`?      | rounds × seed × λ, six GLM families                               | synthetic GLM                     | —                               | `experiments/experimentH/`                |

Headline numbers (final clean / PGD accuracy with mean ± std across 5 seeds)
are tabulated in [Empirical/experiment_main_table.md](Empirical/experiment_main_table.md).
Aggregated per-set CSVs live in `experiments/figures/<set>/`, and per-run
summaries are emitted as `final_summary_<timestamp>.{csv,json}` inside every
algorithm × seed directory.

---

## Reproducing a single run

From `Empirical/`:

```bash
# pure clean training + Secant-BADGE acquisition
python main.py \
  --dataset cifar10 --model resnet18 \
  --acquisition_method ours_badge_secant \
  --initial_labeled_size 500 --acquisition_batch_size 100 --rounds 20 \
  --seed 0

# adversarial training + Secant-BADGE with the p75 gate (paper default)
python main.py \
  --dataset cifar10 --model resnet18 \
  --acquisition_method ours_badge_secant_p75 \
  --train_mode adv --adv_train_attack pgd \
  --adv_train_epsilon 0.00392156862745098 --adv_train_steps 3 \
  --initial_labeled_size 500 --acquisition_batch_size 100 --rounds 20 \
  --seed 0
```

Method strings recognised by `--acquisition_method` map to files in
[Empirical/acquisition/](Empirical/acquisition/) — e.g. `random`, `entropy`,
`badge`, `saal`, `coreset`, `ours_badge_secant{,_p10,_p25,_p50,_p75,_p90}`.

## Reproducing a whole set (Main / A–F)

Each empirical set is described by a JSON in
[Empirical/configs/experiment_sets/](Empirical/configs/experiment_sets/). The
typical workflow is **build → launch → aggregate**:

```bash
cd Empirical

# 1. expand the set JSON into per-(dataset, model, method, seed) commands
python scripts/build_experiment_sets.py \
  --set set_ADV_main \
  --experiments-root ../experiments \
  --jobs-per-gpu 1

# 2. dispatch those commands across all visible GPUs
python scripts/launch_experiment_set.py \
  --set-root ../experiments/set_ADV_main \
  --jobs-per-gpu 1

# 3. roll up per-run final_summary_*.json into per-set CSVs
python scripts/aggregate_experiment_set.py \
  --set-root ../experiments/set_ADV_main
```

To build and run **every** set in one shot:

```bash
python scripts/launch_all_experiments.py \
  --experiments-root ../experiments \
  --jobs-per-gpu 1 \
  --aggregate-after
```

SLURM launchers for cluster runs are checked into
[Empirical/scripts/](Empirical/scripts/) (`run_*_sbatch.sh`).

## Reproducing the GLM theory (G / H)

```bash
python Theoretical/glm_theory/run_glm_theory_experiments.py \
  --glm gaussian logistic softmax poisson gamma exponential \
  --seeds 0 1 2 3 4 \
  --rounds 20 --batch-size 50 --pool-size 2000 \
  --init-size 100 --dim 20 --alpha 75 --eps-acq 0.25 \
  --outdir ../experiments/experimentG
```

`Theoretical/scripts/run_glm_theory_sbatch.sh` and the `_R100` /
`_extension` variants are the SLURM wrappers used to produce
`experimentG/` and `experimentH/`. Plots are regenerated from the resulting
`metrics.csv` by:

```bash
python Theoretical/glm_theory/aggregate_and_plot.py \
  --metrics ../experiments/experimentG/metrics.csv \
  --outdir ../experiments/experimentG/figures
python Theoretical/glm_theory/make_paper_figures.py
```

---

## Outputs

- **Per run** — `seed_<k>/round_<t>/`: round CSVs (`selected_indices/…`,
  per-round metrics), per-epoch training logs, and a final
  `final_summary_<timestamp>.{csv,json}` with `final_clean_acc`,
  `final_pgd_acc`, `final_fgsm_acc`, `best_*`, and timing.
- **Per set** — `experiments/<set>/an_repo_manifest.csv` lists every
  `(dataset, model, method, seed)` cell and its summary file.
- **Aggregated** — figures and summary tables in
  `experiments/figures/<set>/` (Main + A–F) and
  `experiments/experimentG/figures/`, `experiments/experimentH/figures/`.

---

## Requirements

- Python ≥ 3.10, PyTorch with CUDA (single-GPU is fine; multi-GPU is used for
  cross-method dispatch in `launch_experiment_set.py`).
- Empirical: `pip install -r Empirical/requirements.txt`.
- Theoretical: NumPy / SciPy / matplotlib only — CPU is sufficient.

The `experiments/`, `Theoretical/results/`, `data/`, and `Empirical/data/`
directories are git-ignored; only code and aggregated artifacts checked in
under `experiments/figures/` are versioned.

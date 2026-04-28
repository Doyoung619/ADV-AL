# Server Transfer: BADGE/CoreSet Baselines

This branch is prepared for moving only the requested baseline records and rerun commands to another server.

Branch name:

```bash
codex/server-badge-coreset-transfer
```

Latest newly implemented acquisition method name in this codebase:

```bash
ours_secant_logdet_refine
```

The p10 D-norm prefilter variant is:

```bash
ours_secant_logdet_refine_p10
```

Equivalent explicit CLI:

```bash
--acquisition-method ours_secant_logdet_refine \
--prefilter-metric D \
--prefilter-drop-percent 10
```

Here `D(x)=||phi(x)||_2` and `phi(x)=[g_clean(x); g_adv(x)-g_clean(x)]`.

## Included Experiment Records

Only BADGE and CoreSet records from the requested experiment folders are intended to be carried:

```text
experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_badge_b100_r20
experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_coreset_b100_r20

experiments/set_ADV_main_resnet18_cifar100/cifar100_resnet18_badge_b100_r20
experiments/set_ADV_main_resnet18_cifar100/cifar100_resnet18_coreset_b100_r20

experiments/set_ADV_main_smallcnn_cifar10/cifar10_small_cnn_badge_b100_r20
experiments/set_ADV_main_smallcnn_cifar10/cifar10_small_cnn_coreset_b100_r20

experiments/set_ADV_main_smallcnn_cifar100/cifar100_small_cnn_badge_p10_b100_r20
experiments/set_ADV_main_smallcnn_cifar100/cifar100_small_cnn_coreset_p10_b100_r20
```

The `smallcnn_cifar100` directory names contain `_p10` for historical compatibility with the existing local records. The configs inside those folders still use the baseline acquisition methods `badge` and `coreset`.

The common setting is adversarial training:

```text
initial_labeled_size = 500
acquisition_size     = 100
num_rounds           = 20
epochs_per_round     = 80
seeds                = 0,1,2,3,4
train_mode           = adv
adv_train_attack     = pgd
adv_train_epsilon    = 1/255
adv_train_steps      = 3
eval epsilon         = 1/255
```

## Checkout On Another Server

```bash
git fetch origin codex/server-badge-coreset-transfer
git checkout codex/server-badge-coreset-transfer
```

Prepare a Python environment that can run the repository, then point the script at the dataset path:

```bash
export DATA_DIR=/path/to/data
export EXPERIMENTS_ROOT=$PWD/experiments
```

If the dataset is not already present, add:

```bash
export DOWNLOAD_IF_MISSING=1
```

## Generate Commands Only

This writes commands without launching jobs:

```bash
bash scripts/run_server_badge_coreset_baselines.sh --dry-run
```

Output command file:

```text
experiments/server_badge_coreset_commands/commands_all.txt
```

Per-GPU command files:

```text
experiments/server_badge_coreset_commands/commands_gpu_<id>.sh
```

## Run On GPUs

Example using GPUs 0,1,2,3 with one process per GPU:

```bash
export GPU_IDS=0,1,2,3
export JOBS_PER_GPU=1
bash scripts/run_server_badge_coreset_baselines.sh --run
```

Example using only GPU 2:

```bash
export GPU_IDS=2
export JOBS_PER_GPU=1
bash scripts/run_server_badge_coreset_baselines.sh --run
```

The script creates one background worker per generated GPU script and waits for all workers.

## Rerun A Single Job Manually

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset cifar10 \
  --model resnet18 \
  --acquisition-method badge \
  --initial-labeled-size 500 \
  --acquisition-size 100 \
  --num-rounds 20 \
  --epochs-per-round 80 \
  --epsilon 0.00392156862745098 \
  --seed 0 \
  --output-dir experiments/set_ADV_main_resnet18_cifar10/cifar10_resnet18_badge_b100_r20 \
  --run-name seed_0 \
  --train-mode adv \
  --adv-train-attack pgd \
  --adv-train-epsilon 0.00392156862745098 \
  --adv-train-steps 3 \
  --data-dir "$DATA_DIR" \
  --num-workers 8 \
  --pin-memory \
  --skip-logit-mismatch-eval \
  --no-save-checkpoints
```

Change `--acquisition-method` to `coreset`, `--dataset`, `--model`, `--seed`, and `--output-dir` for the other baseline runs.

## Notes For Codex On The Server

- Do not run all of `experiments/`; only the eight BADGE/CoreSet folders listed above are relevant.
- The latest custom method name to remember is `ours_secant_logdet_refine`.
- The current p10 prefilter variant is `ours_secant_logdet_refine_p10`, equivalent to `--prefilter-metric D --prefilter-drop-percent 10`.
- For reproducing the transferred baseline records, only `badge` and `coreset` should be used.

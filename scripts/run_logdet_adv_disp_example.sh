#!/usr/bin/env bash
set -euo pipefail

# Example usage for the adversarial displacement log-det acquisition method.
# Defaults:
# - attack: fgsm
# - epsilon: 1/255
# - lambda: 1e-3

python main.py \
  --dataset cifar10 \
  --model small_cnn \
  --acquisition_method logdet_adv_disp \
  --logdet-adv-disp-attack fgsm \
  --logdet-adv-disp-epsilon 0.0039215686 \
  --logdet-adv-disp-lambda 1e-3 \
  --acquisition_size 50 \
  --num_rounds 20

# PGD variant:
# python main.py \
#   --dataset cifar10 \
#   --model small_cnn \
#   --acquisition_method logdet_adv_disp \
#   --logdet-adv-disp-attack pgd \
#   --logdet-adv-disp-epsilon 0.0039215686 \
#   --logdet-adv-disp-pgd-steps 10 \
#   --logdet-adv-disp-pgd-step-size 0.0007843137 \
#   --logdet-adv-disp-pgd-random-start \
#   --logdet-adv-disp-lambda 1e-3 \
#   --acquisition_size 50 \
#   --num_rounds 20

# Swap-refined variant (greedy + 1-swap local search):
# python main.py \
#   --dataset cifar10 \
#   --model small_cnn \
#   --acquisition_method logdet_adv_disp_swap \
#   --logdet-adv-disp-attack fgsm \
#   --logdet-adv-disp-epsilon 0.0039215686 \
#   --logdet-adv-disp-lambda 1e-3 \
#   --logdet-adv-disp-swap-max-rounds 3 \
#   --logdet-adv-disp-swap-top-unselected 200 \
#   --logdet-adv-disp-swap-top-selected 0 \
#   --acquisition_size 50 \
#   --num_rounds 20

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition import build_acquisition_strategy
from acquisition.secant_badge import compute_secant_badge_embeddings
from acquisition.secant_logdet_refine import (
    compute_clean_grad_norm_scores,
    prefilter_candidates_by_clean_grad_norm,
)


class TinyFeatureNet(nn.Module):
    def __init__(self, num_classes: int = 4, feature_dim: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        h = self.features(x)
        z = self.classifier(h)
        if return_features:
            return z, h
        return z


def _cfg(metric: str = "none", drop_percent: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        seed=123,
        acquisition_method="ours_secant_logdet_refine",
        acquisition_attack="pgd",
        epsilon_acq=0.0,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        logdet_lambda=1e-3,
        logdet_adv_disp_score_chunk_size=128,
        logdet_adv_disp_jitter=1e-8,
        logdet_adv_disp_swap_jitter=1e-8,
        logdet_adv_disp_swap_max_rounds=1,
        logdet_adv_disp_swap_top_unselected=0,
        logdet_adv_disp_swap_top_selected=0,
        logdet_adv_disp_swap_improvement_tol=1e-8,
        prefilter_metric=metric,
        prefilter_drop_percent=drop_percent,
        cifar10_mean=None,
        cifar10_std=None,
    )


def main() -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.rand(10, 3, 8, 8)
    labels = torch.zeros(10, dtype=torch.long)
    indices = torch.arange(100, 110, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=5, shuffle=False)
    model = TinyFeatureNet().to(device)

    phi_zero, parts_zero = compute_secant_badge_embeddings(
        model=model,
        unlabeled_loader=loader,
        device=device,
        epsilon_acq=0.0,
        attack_type="pgd",
        attack_steps=2,
        attack_random_start=False,
        mean=None,
        std=None,
        return_parts=True,
    )
    grad_dim = parts_zero["g_clean"].size(1)
    assert phi_zero.shape == (10, 2 * grad_dim)
    assert parts_zero["g_clean"].shape == (10, grad_dim)
    assert compute_clean_grad_norm_scores(parts_zero["g_clean"]).shape == (10,)
    assert torch.allclose(phi_zero[:, grad_dim:], torch.zeros_like(parts_zero["g_clean"]), atol=1e-6, rtol=1e-6)

    retained, _, debug = prefilter_candidates_by_clean_grad_norm(
        indices=indices.numpy(),
        g_clean=parts_zero["g_clean"],
        drop_percent=10.0,
        budget=4,
    )
    assert len(retained) == 9
    assert debug["drop_count"] == 1
    assert debug["metric"] == "clean_grad_norm"

    no_filter = build_acquisition_strategy("ours_secant_logdet_refine", _cfg("none", 0.0)).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    clean_filter_off = build_acquisition_strategy("ours_secant_logdet_refine", _cfg("clean_grad_norm", 0.0)).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert no_filter.selected_indices.tolist() == clean_filter_off.selected_indices.tolist()
    assert clean_filter_off.extras["candidate_pool_size"] == 10
    assert clean_filter_off.extras["prefilter_enabled"] is False

    clean_filter_p10 = build_acquisition_strategy("ours_secant_logdet_refine", _cfg("clean_grad_norm", 10.0)).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert len(clean_filter_p10.selected_indices) == 4
    assert clean_filter_p10.extras["prefilter_enabled"] is True
    assert clean_filter_p10.extras["prefilter_metric"] == "clean_grad_norm"
    assert clean_filter_p10.extras["candidate_pool_size"] == 9
    print("smoke_test_secant_clean_prefilter: ok")


if __name__ == "__main__":
    main()

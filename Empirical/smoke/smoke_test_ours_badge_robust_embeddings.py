from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition import build_acquisition_strategy
from acquisition.badge import compute_badge_embeddings
from acquisition.secant_badge import (
    compute_clean_grad_norm,
    compute_joint_embedding_norm,
    compute_jointadv_badge_embeddings,
    compute_secant_badge_embeddings,
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


def _cfg(method: str, prefilter_metric: str = "none", prefilter_drop_percent: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        seed=123,
        acquisition_method=method,
        acquisition_attack="pgd",
        epsilon_acq=1.0 / 255.0,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        badge_projection_dim=0,
        badge_candidate_cap=None,
        cifar10_mean=None,
        cifar10_std=None,
        prefilter_metric=prefilter_metric,
        prefilter_drop_percent=prefilter_drop_percent,
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

    badge = compute_badge_embeddings(model=model, unlabeled_loader=loader, device=device)
    grad_dim = badge.size(1)

    phi_secant_zero, secant_parts = compute_secant_badge_embeddings(
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
    assert secant_parts["g_clean"].shape == badge.shape
    assert secant_parts["g_adv"].shape == badge.shape
    assert phi_secant_zero.shape == (badge.size(0), 2 * grad_dim)
    assert torch.allclose(secant_parts["g_clean"], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(secant_parts["g_adv"], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_secant_zero[:, :grad_dim], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_secant_zero[:, grad_dim:], torch.zeros_like(badge), atol=1e-6, rtol=1e-6)
    secant_metric = compute_clean_grad_norm(secant_parts["g_clean"])
    assert secant_metric.shape == (badge.size(0),)
    assert torch.allclose(secant_metric, badge.float().norm(dim=1), atol=1e-6, rtol=1e-6)

    phi_joint_zero, joint_parts = compute_jointadv_badge_embeddings(
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
    assert joint_parts["g_clean"].shape == badge.shape
    assert joint_parts["g_adv"].shape == badge.shape
    assert phi_joint_zero.shape == (badge.size(0), 2 * grad_dim)
    assert torch.allclose(joint_parts["g_clean"], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(joint_parts["g_adv"], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_joint_zero[:, :grad_dim], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_joint_zero[:, grad_dim:], badge, atol=1e-6, rtol=1e-6)
    joint_metric = compute_joint_embedding_norm(joint_parts["g_clean"], joint_parts["g_adv"])
    assert joint_metric.shape == (badge.size(0),)
    assert torch.allclose(joint_metric, phi_joint_zero.float().norm(dim=1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(joint_metric, secant_metric * (2.0 ** 0.5), atol=1e-6, rtol=1e-6)

    for method, embedding_type, filter_metric in [
        ("ours_badge_secant", "secant", "secant_clean_grad_norm"),
        ("ours_badge_jointadv", "jointadv", "joint_embedding_norm"),
    ]:
        strategy = build_acquisition_strategy(method, _cfg(method))
        out = strategy.select(
            model=model,
            unlabeled_loader=loader,
            labeled_loader=None,
            unlabeled_indices=indices.numpy(),
            budget=4,
            device=device,
        )
        assert len(out.selected_indices) == 4
        assert len(np.unique(out.selected_indices)) == 4
        assert out.extras["method"] == method
        assert out.extras["embedding_type"] == embedding_type
        assert out.extras["base_gradient_dim"] == grad_dim
        assert out.extras["embedding_dim"] == 2 * grad_dim
        assert out.extras["prefilter_enabled"] is False

        no_filter_metric_strategy = build_acquisition_strategy(method, _cfg(method, filter_metric, 0.0))
        no_filter_metric_out = no_filter_metric_strategy.select(
            model=model,
            unlabeled_loader=loader,
            labeled_loader=None,
            unlabeled_indices=indices.numpy(),
            budget=4,
            device=device,
        )
        assert np.array_equal(out.selected_indices, no_filter_metric_out.selected_indices)
        assert no_filter_metric_out.extras["prefilter_enabled"] is False

        p10_strategy = build_acquisition_strategy(method, _cfg(method, filter_metric, 10.0))
        p10_out = p10_strategy.select(
            model=model,
            unlabeled_loader=loader,
            labeled_loader=None,
            unlabeled_indices=indices.numpy(),
            budget=4,
            device=device,
        )
        assert len(p10_out.selected_indices) == 4
        assert len(np.unique(p10_out.selected_indices)) == 4
        assert p10_out.extras["prefilter_enabled"] is True
        assert p10_out.extras["prefilter_metric"] == filter_metric
        assert p10_out.extras["prefilter_pool_size_before"] == 10
        assert p10_out.extras["prefilter_pool_size_after"] == 9

    print("smoke_test_ours_badge_robust_embeddings: ok")


if __name__ == "__main__":
    main()

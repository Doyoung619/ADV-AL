from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition import build_acquisition_strategy
from acquisition.badge import compute_badge_embeddings
from acquisition.secant_badge import compute_secant_badge_embeddings


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


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        seed=123,
        acquisition_method="ours_secant_badge",
        acquisition_attack="pgd",
        epsilon_acq=1.0 / 255.0,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        badge_projection_dim=0,
        badge_candidate_cap=None,
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
    badge = compute_badge_embeddings(model=model, unlabeled_loader=loader, device=device)
    grad_dim = badge.size(1)
    assert parts_zero["g_clean"].shape == badge.shape
    assert parts_zero["g_adv"].shape == badge.shape
    assert phi_zero.shape == (badge.size(0), 2 * grad_dim)
    assert torch.allclose(parts_zero["g_clean"], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(parts_zero["g_adv"], parts_zero["g_clean"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_zero[:, :grad_dim], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_zero[:, grad_dim:], torch.zeros_like(badge), atol=1e-6, rtol=1e-6)

    cfg = _cfg()
    strategy = build_acquisition_strategy("ours_secant_badge", cfg)
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
    assert out.extras["method"] == "ours_secant_badge"
    assert out.extras["base_gradient_dim"] == grad_dim
    assert out.extras["embedding_dim"] == 2 * grad_dim
    print("smoke_test_ours_secant_badge: ok")
    print("selected_indices:", out.selected_indices.tolist())


if __name__ == "__main__":
    main()

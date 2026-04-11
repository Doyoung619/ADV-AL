from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition.logdet_adv_disp import (
    AdvGradDisplacementLogDetStrategy,
    build_adv_gradient_displacement_gram,
    compute_adv_gradient_displacement_components,
)


class TinyFeatureNet(nn.Module):
    def __init__(self, num_classes: int = 3, feature_dim: int = 8):
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
        pool_batch_size=5,
        cifar10_mean=None,
        cifar10_std=None,
        epsilon_acq=1.0 / 255.0,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        logdet_lambda=1e-3,
        logdet_adv_disp_lambda=1e-3,
        logdet_adv_disp_score_chunk_size=4,
        logdet_adv_disp_jitter=1e-8,
        adv_grad_displacement_use_explicit_embedding=False,
        debug_save_adv_scores=False,
    )


def main() -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.rand(12, 3, 8, 8)
    labels = torch.zeros(12, dtype=torch.long)
    indices = torch.arange(100, 112, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=5, shuffle=False)
    model = TinyFeatureNet().to(device)
    cfg = _cfg()

    components = compute_adv_gradient_displacement_components(
        model=model,
        unlabeled_loader=loader,
        epsilon=cfg.epsilon_acq,
        attack_steps=cfg.attack_steps,
        attack_step_size=cfg.attack_step_size,
        attack_random_start=cfg.attack_random_start,
        mean=cfg.cifar10_mean,
        std=cfg.cifar10_std,
        device=device,
        tensor_batch_size=cfg.pool_batch_size,
    )
    gram = build_adv_gradient_displacement_gram(
        components=components,
        device=device,
        score_chunk_size=cfg.logdet_adv_disp_score_chunk_size,
    )
    assert torch.isfinite(gram).all(), "Gram contains non-finite values"
    assert torch.allclose(gram, gram.t(), atol=1e-5, rtol=1e-5), "Gram is not symmetric"

    strategy = AdvGradDisplacementLogDetStrategy(cfg)
    out1 = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    out2 = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert len(out1.selected_indices) == 4
    assert len(np.unique(out1.selected_indices)) == 4
    assert np.array_equal(out1.selected_indices, out2.selected_indices), "Selection should be deterministic"
    assert np.isfinite(out1.scores).all(), "Scores contain non-finite values"
    assert out1.extras["validation"]["gram_symmetric"]
    assert out1.extras["validation"]["selected_unique"]
    assert out1.extras["validation"]["selected_count_matches_budget"]
    print("smoke_test_adv_grad_displacement_logdet: ok")
    print("selected_indices:", out1.selected_indices.tolist())
    print("selector_mode:", out1.extras["selector_mode"])


if __name__ == "__main__":
    main()

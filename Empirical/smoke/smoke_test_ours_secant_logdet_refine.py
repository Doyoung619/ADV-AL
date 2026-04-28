from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition import build_acquisition_strategy
from acquisition.badge import compute_badge_embeddings
from acquisition.logdet_refine import build_gram_matrix, forward_greedy_select, refine_by_swap
from acquisition.secant_badge import compute_secant_badge_embeddings
from acquisition.secant_logdet_refine import compute_secant_norm_scores, prefilter_by_secant_norm


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
        acquisition_method="ours_secant_logdet_refine",
        acquisition_attack="pgd",
        epsilon_acq=1.0 / 255.0,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        logdet_lambda=1e-3,
        logdet_adv_disp_lambda=1e-3,
        logdet_adv_disp_score_chunk_size=128,
        logdet_adv_disp_jitter=1e-8,
        logdet_adv_disp_swap_jitter=1e-8,
        logdet_adv_disp_swap_max_rounds=3,
        logdet_adv_disp_swap_top_unselected=0,
        logdet_adv_disp_swap_top_selected=0,
        logdet_adv_disp_swap_improvement_tol=1e-8,
        prefilter_metric="none",
        prefilter_drop_percent=0.0,
        badge_projection_dim=0,
        badge_candidate_cap=None,
        cifar10_mean=None,
        cifar10_std=None,
    )


def main() -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.rand(12, 3, 8, 8)
    labels = torch.zeros(12, dtype=torch.long)
    indices = torch.arange(200, 212, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=4, shuffle=False)
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
    assert torch.allclose(phi_zero[:, :grad_dim], badge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phi_zero[:, grad_dim:], torch.zeros_like(badge), atol=1e-6, rtol=1e-6)
    d_zero = compute_secant_norm_scores(phi_zero)
    assert d_zero.shape == (phi_zero.size(0),)
    assert torch.allclose(d_zero, parts_zero["g_clean"].norm(dim=1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        d_zero.pow(2),
        parts_zero["g_clean"].pow(2).sum(dim=1) + parts_zero["correction"].pow(2).sum(dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    local_all, phi_all, d_all, dbg_all = prefilter_by_secant_norm(phi_zero, drop_percent=0.0, budget=5)
    assert np.array_equal(local_all, np.arange(phi_zero.size(0)))
    assert torch.equal(phi_all, phi_zero)
    assert torch.allclose(d_all, d_zero)
    assert not dbg_all["enabled"]
    local_p10, phi_p10, _, dbg_p10 = prefilter_by_secant_norm(phi_zero, drop_percent=10.0, budget=5)
    assert len(local_p10) == int(np.ceil(0.9 * phi_zero.size(0)))
    assert phi_p10.shape[0] == len(local_p10)
    assert dbg_p10["drop_count"] == phi_zero.size(0) - len(local_p10)

    kernel = build_gram_matrix(phi_zero, chunk_size=4, dtype=torch.float32, device=torch.device("cpu"))
    picked_forward, forward_debug = forward_greedy_select(
        kernel=kernel,
        query_size=5,
        lambda_reg=1e-3,
        score_chunk_size=8,
        ambient_dim=phi_zero.size(1),
    )
    objectives = forward_debug["forward_objectives"]
    assert all(objectives[i + 1] >= objectives[i] - 1e-8 for i in range(len(objectives) - 1))

    picked_refined, swap_debug = refine_by_swap(
        kernel=kernel,
        selected_local=picked_forward,
        lambda_reg=1e-3,
        score_chunk_size=8,
        ambient_dim=phi_zero.size(1),
        max_swap_rounds=3,
        swap_top_unselected=0,
        swap_top_selected=0,
    )
    assert len(picked_refined) == len(picked_forward)
    assert len(np.unique(picked_refined)) == len(picked_refined)
    assert swap_debug["final_logdet_objective"] >= forward_debug["final_logdet_objective"] - 1e-8

    g = torch.Generator().manual_seed(0)
    synthetic = torch.randn(14, 5, generator=g)
    synthetic[8:11] = synthetic[0:3] + 0.05 * torch.randn(3, 5, generator=g)
    synthetic_kernel = build_gram_matrix(synthetic, chunk_size=14, dtype=torch.float32, device=torch.device("cpu"))
    synthetic_forward, synthetic_forward_debug = forward_greedy_select(
        kernel=synthetic_kernel,
        query_size=5,
        lambda_reg=0.2,
        ambient_dim=synthetic.size(1),
    )
    synthetic_refined, synthetic_swap_debug = refine_by_swap(
        kernel=synthetic_kernel,
        selected_local=synthetic_forward,
        lambda_reg=0.2,
        ambient_dim=synthetic.size(1),
        max_swap_rounds=5,
        swap_top_unselected=0,
        swap_top_selected=0,
    )
    assert len(synthetic_refined) == len(synthetic_forward)
    assert synthetic_swap_debug["accepted_swaps"] >= 1
    assert synthetic_swap_debug["final_logdet_objective"] > synthetic_forward_debug["final_logdet_objective"]

    cfg = _cfg()
    strategy = build_acquisition_strategy("ours_secant_logdet_refine", cfg)
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
    assert out.extras["method"] == "ours_secant_logdet_refine"
    assert out.extras["base_gradient_dim"] == grad_dim
    assert out.extras["embedding_dim"] == 2 * grad_dim
    assert out.extras["refinement_non_decrease"]
    assert not out.extras["prefilter_enabled"]
    assert out.extras["refined_final_logdet_objective"] >= out.extras["forward_final_logdet_objective"] - 1e-8
    cfg_d0 = _cfg()
    cfg_d0.prefilter_metric = "D"
    cfg_d0.prefilter_drop_percent = 0.0
    out_d0 = build_acquisition_strategy("ours_secant_logdet_refine", cfg_d0).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert np.array_equal(out_d0.selected_indices, out.selected_indices)
    assert out_d0.extras["forward_objectives"] == out.extras["forward_objectives"]
    assert not out_d0.extras["prefilter_enabled"]
    cfg_p10 = _cfg()
    cfg_p10.prefilter_metric = "D"
    cfg_p10.prefilter_drop_percent = 10.0
    out_p10 = build_acquisition_strategy("ours_secant_logdet_refine", cfg_p10).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert len(out_p10.selected_indices) == 4
    assert len(np.unique(out_p10.selected_indices)) == 4
    assert out_p10.extras["prefilter_enabled"]
    assert out_p10.extras["prefilter_metric"] == "D"
    assert out_p10.extras["prefilter_pool_size_after"] == int(np.ceil(0.9 * len(indices)))
    assert out_p10.extras["refinement_non_decrease"]
    cfg_suffix = _cfg()
    suffix_strategy = build_acquisition_strategy("ours_secant_logdet_refine_p10", cfg_suffix)
    assert suffix_strategy.cfg.prefilter_metric == "D"
    assert suffix_strategy.cfg.prefilter_drop_percent == 10.0
    print("smoke_test_ours_secant_logdet_refine: ok")
    print("selected_indices:", out.selected_indices.tolist())
    print("selected_indices_p10:", out_p10.selected_indices.tolist())
    print("refinement_improvement:", out.extras["refinement_improvement"])


if __name__ == "__main__":
    main()

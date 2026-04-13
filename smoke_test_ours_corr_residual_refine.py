from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acquisition import build_acquisition_strategy
from acquisition.corr_residual_refine import (
    _mgs_basis_rows,
    _residual_from_basis,
    clean_gradient_percentile_gate,
    compute_corr_residual_embeddings,
    refine_by_residual_swaps,
    residual_forward_select,
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


def _cfg(epsilon_acq: float = 1.0 / 255.0, clean_gate_percentile: float = 10.0) -> SimpleNamespace:
    return SimpleNamespace(
        seed=123,
        acquisition_method="ours_corr_residual_refine",
        acquisition_attack="pgd",
        epsilon_acq=epsilon_acq,
        attack_steps=2,
        attack_step_size=None,
        attack_random_start=False,
        clean_gate_percentile=clean_gate_percentile,
        corr_residual_refine_max_rounds=2,
        corr_residual_refine_incoming_shortlist=4,
        corr_residual_refine_outgoing_shortlist=2,
        corr_residual_refine_improvement_tol=1e-10,
        corr_residual_score_chunk_size=8,
        tiny=1e-8,
        cifar10_mean=None,
        cifar10_std=None,
    )


def main() -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.rand(12, 3, 8, 8)
    labels = torch.zeros(12, dtype=torch.long)
    indices = torch.arange(300, 312, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=4, shuffle=False)
    model = TinyFeatureNet().to(device)

    parts_zero = compute_corr_residual_embeddings(
        model=model,
        unlabeled_loader=loader,
        device=device,
        epsilon_acq=0.0,
        attack_type="pgd",
        attack_steps=2,
        attack_random_start=False,
        mean=None,
        std=None,
    )
    g_clean = parts_zero["g_clean"]
    gamma = parts_zero["gamma"]
    assert g_clean.shape == gamma.shape
    assert g_clean.ndim == 2
    assert torch.allclose(gamma, torch.zeros_like(gamma), atol=1e-6, rtol=1e-6)
    assert parts_zero["clean_norm"].shape == (12,)

    unique_scores = torch.arange(10, dtype=torch.float32)
    local_all, gate_all = clean_gradient_percentile_gate(unique_scores, drop_percentile=0.0, budget=4)
    assert np.array_equal(local_all, np.arange(10))
    assert gate_all["enabled"] is False
    local_p10, gate_p10 = clean_gradient_percentile_gate(unique_scores, drop_percentile=10.0, budget=4)
    assert np.array_equal(local_p10, np.arange(1, 10))
    assert gate_p10["actual_drop_count"] == 1
    assert gate_p10["tau"] == 1.0
    local_p90_budget, gate_p90_budget = clean_gradient_percentile_gate(unique_scores, drop_percentile=90.0, budget=4)
    assert len(local_p90_budget) == 4
    assert gate_p90_budget["tau"] == 6.0

    synthetic = torch.eye(5, dtype=torch.float32)
    v = synthetic[0].to(dtype=torch.float64)
    empty_q = _mgs_basis_rows(synthetic[:0])
    assert torch.allclose(_residual_from_basis(empty_q, v), v)
    q_span = _mgs_basis_rows(synthetic[:1])
    assert _residual_from_basis(q_span, v).norm().item() <= 1e-10

    clean_norms = torch.linspace(1.0, 2.0, steps=5)
    selected_forward, q_forward, forward_debug = residual_forward_select(
        gamma=synthetic,
        clean_norms=clean_norms,
        v_target=v,
        budget=3,
        score_chunk_size=3,
    )
    assert selected_forward[0] == 0
    assert q_forward.shape[0] >= 1
    assert forward_debug["forward_objectives"][-1] >= forward_debug["forward_objectives"][0] - 1e-10

    refined, refine_debug = refine_by_residual_swaps(
        gamma=synthetic,
        clean_norms=clean_norms,
        selected_local=selected_forward,
        v_target=v,
        max_rounds=2,
        incoming_shortlist=0,
        outgoing_shortlist=0,
    )
    assert len(refined) == len(selected_forward)
    assert refine_debug["final_objective"] >= refine_debug["initial_objective"] - 1e-8

    out_zero = build_acquisition_strategy("ours_corr_residual_refine", _cfg(epsilon_acq=0.0, clean_gate_percentile=10.0)).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert len(out_zero.selected_indices) == 4
    assert len(np.unique(out_zero.selected_indices)) == 4
    assert out_zero.extras["zero_gamma_fallback"] is True
    assert out_zero.extras["clean_gate_enabled"] is True
    assert out_zero.extras["clean_gate_actual_drop_count"] == 1
    assert out_zero.extras["g_clean_shape"] == list(g_clean.shape)
    assert out_zero.extras["gamma_shape"] == list(gamma.shape)
    assert out_zero.extras["v_R_shape"] == [g_clean.size(1)]

    out = build_acquisition_strategy("ours_corr_residual_refine_p10", _cfg(epsilon_acq=1.0 / 255.0, clean_gate_percentile=0.0)).select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=indices.numpy(),
        budget=4,
        device=device,
    )
    assert len(out.selected_indices) == 4
    assert len(np.unique(out.selected_indices)) == 4
    assert out.extras["method"] == "ours_corr_residual_refine"
    assert out.extras["clean_gate_percentile"] == 10.0
    assert out.extras["refined_final_objective"] >= out.extras["refinement_initial_objective"] - 1e-8
    print("smoke_test_ours_corr_residual_refine: ok")
    print("selected_indices_zero_gamma:", out_zero.selected_indices.tolist())
    print("selected_indices:", out.selected_indices.tolist())
    print("refinement_swaps:", out.extras["refinement_accepted_swaps"])


if __name__ == "__main__":
    main()

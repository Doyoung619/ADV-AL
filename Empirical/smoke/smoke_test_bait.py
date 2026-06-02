import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from acquisition import build_acquisition_strategy
from acquisition.bait import (
    accumulate_fisher_from_factors,
    bait_backward_prune,
    bait_forward_greedy,
    build_class_cov_factors,
    build_classification_fisher_factor,
    downdate_inverse_with_factor,
    update_inverse_with_factor,
    woodbury_trace_gain,
    woodbury_trace_increase_downdate,
)
from config import parse_config


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


class TinyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dtype = torch.float64

    # Test 1: Fisher factor correctness V V^T == explicit Fisher (class-major vectorization).
    feat = torch.randn(5, dtype=dtype)
    prob = F.softmax(torch.randn(3, dtype=dtype), dim=0)
    cov_factor = build_class_cov_factors(prob.unsqueeze(0))[0]
    V = build_classification_fisher_factor(feat, cov_factor)
    fisher_from_factor = V @ V.t()
    class_cov = torch.diag(prob) - torch.outer(prob, prob)
    fisher_explicit = torch.kron(class_cov, torch.outer(feat, feat))
    assert torch.allclose(fisher_from_factor, fisher_explicit, atol=1e-8, rtol=1e-6), "Fisher factorization mismatch."

    # Test 2 + 3: forward oversampling to 2B and backward pruning to B.
    num_unlabeled = 9
    num_labeled = 6
    feature_dim = 4
    num_classes = 3
    budget = 2
    oversample = 2 * budget

    unlabeled_features = torch.randn(num_unlabeled, feature_dim, dtype=dtype)
    unlabeled_probs = F.softmax(torch.randn(num_unlabeled, num_classes, dtype=dtype), dim=1)
    labeled_features = torch.randn(num_labeled, feature_dim, dtype=dtype)
    labeled_probs = F.softmax(torch.randn(num_labeled, num_classes, dtype=dtype), dim=1)

    unlabeled_cov_factors = build_class_cov_factors(unlabeled_probs)
    labeled_cov_factors = build_class_cov_factors(labeled_probs)
    I_u = accumulate_fisher_from_factors(unlabeled_features, unlabeled_cov_factors)
    I_l = accumulate_fisher_from_factors(labeled_features, labeled_cov_factors)
    M0 = I_l + 1.0 * torch.eye(I_u.size(0), dtype=dtype)
    M0_inv = torch.linalg.inv(M0)

    selected_forward, M_inv_forward, _ = bait_forward_greedy(
        features=unlabeled_features,
        class_cov_factors=unlabeled_cov_factors,
        M_inv=M0_inv,
        I_u=I_u,
        oversample_count=oversample,
        progress_logger=None,
    )
    assert len(selected_forward) == oversample, f"Forward selected {len(selected_forward)} instead of {oversample}."
    assert len(np.unique(np.asarray(selected_forward))) == oversample, "Forward selected indices are not unique."

    selected_final, _, _ = bait_backward_prune(
        features=unlabeled_features,
        class_cov_factors=unlabeled_cov_factors,
        forward_selected=selected_forward,
        target_budget=budget,
        M_inv=M_inv_forward,
        I_u=I_u,
        progress_logger=None,
    )
    assert len(selected_final) == budget, f"Backward kept {len(selected_final)} instead of {budget}."
    assert len(np.unique(np.asarray(selected_final))) == budget, "Backward selected indices are not unique."
    assert set(selected_final).issubset(set(selected_forward)), "Backward result must be subset of forward set."

    # Test 4: Woodbury update/downdate consistency against direct inversion.
    dim = 8
    rank = 3
    A = torch.randn(dim, dim, dtype=dtype)
    M = A @ A.t() + 0.5 * torch.eye(dim, dtype=dtype)
    M_inv = torch.linalg.inv(M)
    V_test = torch.randn(dim, rank, dtype=dtype)

    _, U_upd, Ainv_upd = woodbury_trace_gain(M_inv=M_inv, I_u=torch.eye(dim, dtype=dtype), factor=V_test)
    M_inv_updated = update_inverse_with_factor(M_inv, U_upd, Ainv_upd)
    M_inv_direct = torch.linalg.inv(M + V_test @ V_test.t())
    assert torch.allclose(M_inv_updated, M_inv_direct, atol=1e-8, rtol=1e-6), "Woodbury update mismatch."

    down = woodbury_trace_increase_downdate(
        M_inv=M_inv_updated,
        I_u=torch.eye(dim, dtype=dtype),
        factor=V_test,
    )
    assert down is not None, "Downdate terms unexpectedly invalid."
    _, U_down, Ainv_down = down
    M_inv_recovered = downdate_inverse_with_factor(M_inv_updated, U_down, Ainv_down)
    assert torch.allclose(M_inv_recovered, M_inv, atol=1e-8, rtol=1e-6), "Woodbury downdate mismatch."

    # Test 5: registry + CLI + end-to-end strategy select.
    cfg = parse_config(
        [
            "--acq_method",
            "bait",
            "--bait_lambda",
            "1.0",
            "--bait_oversample_factor",
            "2",
            "--bait_use_bias",
            "true",
        ]
    )
    strategy = build_acquisition_strategy("bait", cfg)

    x_unl = torch.randn(8, 3, 4, 4, dtype=torch.float32)
    x_lab = torch.randn(5, 3, 4, 4, dtype=torch.float32)
    unl_loader = DataLoader(TinyIndexedDataset(x_unl), batch_size=4, shuffle=False, num_workers=0)
    lab_loader = DataLoader(TinyIndexedDataset(x_lab), batch_size=4, shuffle=False, num_workers=0)
    model = TinyNet(input_dim=3 * 4 * 4, hidden_dim=6, num_classes=3).to(torch.device("cpu"))
    model.eval()

    out = strategy.select(
        model=model,
        unlabeled_loader=unl_loader,
        labeled_loader=lab_loader,
        unlabeled_indices=np.arange(len(x_unl), dtype=np.int64),
        budget=2,
        device=torch.device("cpu"),
        progress_logger=None,
    )
    assert len(out.selected_indices) == 2, "End-to-end BAIT selection returned wrong batch size."
    assert len(np.unique(out.selected_indices)) == 2, "End-to-end BAIT selection has duplicate indices."
    assert set(out.selected_indices.tolist()).issubset(set(range(len(x_unl)))), "Selection contains invalid indices."
    assert math.isfinite(float(out.scoring_time_sec))
    assert math.isfinite(float(out.selection_time_sec))

    print("BAIT smoke tests passed.")


if __name__ == "__main__":
    main()

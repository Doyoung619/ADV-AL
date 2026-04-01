import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from acquisition import build_acquisition_strategy
from acquisition.batchbald import (
    batchbald_greedy_select,
    compute_bald_scores_from_mc_probs,
    get_mc_predictive_probs,
)
from config import parse_config


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


class TinyDropoutNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    eps = 1e-12

    # Test 1: shape [N, K, C] for predictive probabilities.
    x = torch.randn(7, 3, 4, 4, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    model = TinyDropoutNet(input_dim=3 * 4 * 4, hidden_dim=12, num_classes=4).to(device)
    model.eval()
    probs_nkc = get_mc_predictive_probs(
        model=model,
        unlabeled_loader=loader,
        mc_passes=6,
        device=device,
        eps=eps,
        progress_logger=None,
    )
    assert probs_nkc.shape == (7, 6, 4), f"Expected [7, 6, 4], got {tuple(probs_nkc.shape)}"

    # Test 2: BatchBALD with batch size 1 should match BALD argmax.
    rnd_logits = torch.randn(11, 9, 3, dtype=torch.float64)
    rnd_probs = F.softmax(rnd_logits, dim=2)
    bald_scores = compute_bald_scores_from_mc_probs(rnd_probs, eps=eps)
    expected = int(torch.argmax(bald_scores).item())
    selected_b1, _ = batchbald_greedy_select(
        probs_nkc=rnd_probs,
        batch_size=1,
        num_joint_entropy_samples=2000,
        exact_joint_entropy_steps=4,
        eps=eps,
        seed=0,
        progress_logger=None,
    )
    assert int(selected_b1[0]) == expected, "BatchBALD(batch_size=1) does not match BALD ranking."

    # Test 3: no duplicates and exact batch length.
    selected_b3, _ = batchbald_greedy_select(
        probs_nkc=rnd_probs,
        batch_size=3,
        num_joint_entropy_samples=2000,
        exact_joint_entropy_steps=2,
        eps=eps,
        seed=0,
        progress_logger=None,
    )
    assert len(selected_b3) == 3, f"Expected 3 selections, got {len(selected_b3)}"
    assert len(np.unique(selected_b3)) == 3, "BatchBALD selected duplicate indices."

    # Test 4: duplicate BALD candidates -> BatchBALD should prefer less redundant second pick.
    custom = torch.tensor(
        [
            # candidate 0
            [[0.999, 0.001], [0.999, 0.001], [0.001, 0.999], [0.001, 0.999]],
            # candidate 1 (near-duplicate of candidate 0)
            [[0.999, 0.001], [0.999, 0.001], [0.001, 0.999], [0.001, 0.999]],
            # candidate 2 (complementary information pattern)
            [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]],
            # candidate 3 (uninformative)
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=torch.float64,
    )  # [N=4, K=4, C=2]
    naive_top2 = torch.topk(compute_bald_scores_from_mc_probs(custom, eps=eps), k=2, largest=True).indices.cpu().numpy()
    batchbald_top2, _ = batchbald_greedy_select(
        probs_nkc=custom,
        batch_size=2,
        num_joint_entropy_samples=5000,
        exact_joint_entropy_steps=4,
        eps=eps,
        seed=0,
        progress_logger=None,
    )
    assert set(naive_top2.tolist()) == {0, 1}, "Naive top-2 BALD is not the expected redundant pair."
    assert 2 in set(batchbald_top2.tolist()), "BatchBALD did not pick the complementary candidate."

    # Test 5: end-to-end strategy smoke.
    cfg = parse_config(
        [
            "--acq_method",
            "batchbald",
            "--batchbald_num_mc_samples",
            "8",
            "--batchbald_num_joint_entropy_samples",
            "3000",
            "--batchbald_exact_joint_entropy_steps",
            "2",
        ]
    )
    strategy = build_acquisition_strategy("batchbald", cfg)
    out = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=np.arange(len(ds), dtype=np.int64),
        budget=3,
        device=device,
        progress_logger=None,
    )
    assert len(out.selected_indices) == 3, f"Expected 3 selected indices, got {len(out.selected_indices)}"
    assert len(np.unique(out.selected_indices)) == 3, "End-to-end BatchBALD selected duplicates."
    assert set(out.selected_indices.tolist()).issubset(set(range(len(ds))))
    assert math.isfinite(float(out.scoring_time_sec))
    assert math.isfinite(float(out.selection_time_sec))

    print("BatchBALD smoke tests passed.")


if __name__ == "__main__":
    main()

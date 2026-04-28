import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from acquisition.margin import MarginStrategy, score_margin


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Test 1: shape test for margin computation from logits
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [1.0, 0.9, 0.2],
            [-0.5, 0.2, 1.2],
        ],
        dtype=torch.float32,
    )
    probs = F.softmax(logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    assert margins.shape == (3,), f"Expected shape [3], got {tuple(margins.shape)}"

    # Test 2: known-value ranking
    # [0.7, 0.2, 0.1] margin=0.5
    # [0.4, 0.35, 0.25] margin=0.05
    probs_known = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]], dtype=torch.float32)
    top2_known = torch.topk(probs_known, k=2, dim=1).values
    margins_known = top2_known[:, 0] - top2_known[:, 1]
    assert torch.allclose(margins_known, torch.tensor([0.5, 0.05])), "Known-value margins mismatch."
    # score=-margin => sample 1 should have larger score (more informative).
    scores_known = -margins_known
    assert scores_known[1] > scores_known[0], "Smaller margin sample is not ranked higher."

    # Build tiny data/model for end-to-end strategy test
    x = torch.randn(10, 3, 8, 8, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).to("cpu")
    model.eval()

    # Test 3: top-k direction (strategy must select smallest-margin samples)
    with torch.no_grad():
        direct_scores = score_margin(model=model, unlabeled_loader=loader, device=torch.device("cpu"))
    expected_local = torch.topk(direct_scores, k=4, largest=True).indices.cpu().numpy()

    cfg = SimpleNamespace()
    strategy = MarginStrategy(cfg)
    out = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=np.arange(len(ds), dtype=np.int64),
        budget=4,
        device=torch.device("cpu"),
        progress_logger=None,
    )
    selected = out.selected_indices
    assert np.array_equal(np.sort(selected), np.sort(expected_local)), "Top-k direction is incorrect for margin."

    # Test 4: uniqueness and length
    assert len(selected) == 4, f"Expected 4 selected indices, got {len(selected)}"
    assert len(np.unique(selected)) == 4, "Selected indices are not unique."

    assert math.isfinite(float(out.scoring_time_sec))
    assert math.isfinite(float(out.selection_time_sec))
    print("Margin smoke tests passed.")


if __name__ == "__main__":
    main()


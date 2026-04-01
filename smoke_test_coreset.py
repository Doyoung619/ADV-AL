import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from acquisition import build_acquisition_strategy
from acquisition.coreset import extract_penultimate_features, kcenter_greedy
from config import parse_config


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


class TinyFeatureNet(nn.Module):
    def __init__(self, input_dim: int, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, feat_dim), nn.ReLU())
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Test 1: shape / feature extraction.
    x = torch.randn(9, 3, 4, 4, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    model = TinyFeatureNet(input_dim=3 * 4 * 4, feat_dim=7, num_classes=4).to(torch.device("cpu"))
    model.eval()
    feats, idxs = extract_penultimate_features(model=model, loader=loader, device=torch.device("cpu"))
    assert feats.shape == (9, 7), f"Expected feature shape (9,7), got {tuple(feats.shape)}"
    assert idxs.shape == (9,), f"Expected indices shape (9,), got {tuple(idxs.shape)}"

    # Test 2: k-Center greedy uniqueness.
    unl = torch.randn(12, 5, dtype=torch.float32)
    lab = torch.randn(4, 5, dtype=torch.float32)
    selected, debug = kcenter_greedy(unlabeled_features=unl, labeled_features=lab, budget=4, chunk_size=64)
    assert len(selected) == 4, f"Expected 4 selected points, got {len(selected)}"
    assert len(np.unique(selected)) == 4, "k-Center selected duplicate indices."
    assert isinstance(debug, dict)

    # Test 3: labeled-set initialization correctness.
    toy_unl = torch.tensor([[0.1], [0.2], [10.0], [11.0]], dtype=torch.float32)
    toy_lab = torch.tensor([[0.0]], dtype=torch.float32)
    sel_toy, _ = kcenter_greedy(unlabeled_features=toy_unl, labeled_features=toy_lab, budget=1, chunk_size=16)
    assert int(sel_toy[0]) in {2, 3}, "Core-set did not prioritize farthest point from labeled centers."

    # Test 4: monotonic distance reduction.
    _, dbg = kcenter_greedy(unlabeled_features=unl, labeled_features=lab, budget=5, chunk_size=64)
    history = dbg["cover_radius_history"]
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-7, "Cover radius should be non-increasing."

    # Test 5: end-to-end smoke test.
    x_unl = torch.randn(10, 3, 4, 4, dtype=torch.float32)
    x_lab = torch.randn(6, 3, 4, 4, dtype=torch.float32)
    unl_loader = DataLoader(TinyIndexedDataset(x_unl), batch_size=4, shuffle=False, num_workers=0)
    lab_loader = DataLoader(TinyIndexedDataset(x_lab), batch_size=4, shuffle=False, num_workers=0)
    cfg = parse_config(["--acq_method", "coreset"])
    strategy = build_acquisition_strategy("coreset", cfg)
    out = strategy.select(
        model=model,
        unlabeled_loader=unl_loader,
        labeled_loader=lab_loader,
        unlabeled_indices=np.arange(len(x_unl), dtype=np.int64),
        budget=3,
        device=torch.device("cpu"),
        progress_logger=None,
    )
    assert len(out.selected_indices) == 3, f"Expected 3 selected indices, got {len(out.selected_indices)}"
    assert len(np.unique(out.selected_indices)) == 3, "Selected indices are not unique."
    assert set(out.selected_indices.tolist()).issubset(set(range(len(x_unl))))
    assert math.isfinite(float(out.scoring_time_sec))
    assert math.isfinite(float(out.selection_time_sec))

    print("CoreSet smoke tests passed.")


if __name__ == "__main__":
    main()

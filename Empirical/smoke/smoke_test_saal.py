import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from acquisition.saal import SAALStrategy, score_saal, score_single_saal


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        # unlabeled label is not used by SAAL; keep placeholder.
        return self.x[idx], 0, int(idx)


def build_tiny_model(in_dim: int, num_classes: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    x = torch.randn(8, 3, 8, 8, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = build_tiny_model(in_dim=3 * 8 * 8, num_classes=4).to(device)
    model.eval()
    params = [p for p in model.parameters()]

    # Test 1: parameter restoration
    before = [p.detach().clone() for p in params]
    _score, _debug = score_single_saal(
        model=model,
        x=x[0].to(device),
        params=params,
        rho=0.05,
        norm="linf",
        return_debug=True,
    )
    for b, p in zip(before, params):
        assert torch.allclose(b, p.detach(), atol=1e-8, rtol=0.0), "Model parameter changed after SAAL scoring."

    # Test 2: score finiteness
    scores = score_saal(
        model=model,
        unlabeled_loader=loader,
        rho=0.05,
        norm="linf",
        device=device,
        progress_logger=None,
    )
    assert torch.isfinite(scores).all().item(), "SAAL scores contain NaN/Inf."

    # Test 3: pseudo-label consistency (must use original pseudo-label)
    score, debug = score_single_saal(
        model=model,
        x=x[1].to(device),
        params=params,
        rho=0.05,
        norm="linf",
        return_debug=True,
    )
    _ = score
    assert debug is not None
    assert debug["pseudo_label_used"] == debug["pseudo_label_original"], "Pseudo-label drifted from original logits."

    # Test 4: rho=0 should match unperturbed CE with pseudo-label
    score_rho0, debug_rho0 = score_single_saal(
        model=model,
        x=x[2].to(device),
        params=params,
        rho=0.0,
        norm="linf",
        return_debug=True,
    )
    assert debug_rho0 is not None
    assert abs(score_rho0 - debug_rho0["clean_loss"]) < 1e-6, "rho=0 does not match clean pseudo-label CE."

    # Test 5: top-k shape and uniqueness
    cfg = SimpleNamespace(
        saal_rho=0.05,
        saal_norm="linf",
        saal_use_kmeanspp=False,
        saal_batchwise_perturb=False,
        candidate_ratio=10.0,
        seed=0,
    )
    strategy = SAALStrategy(cfg)
    out = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=np.arange(len(ds), dtype=np.int64),
        budget=3,
        device=device,
        progress_logger=None,
    )
    sel = out.selected_indices
    assert len(sel) == 3, f"Expected 3 selections, got {len(sel)}"
    assert len(np.unique(sel)) == 3, "Selected indices are not unique."
    assert set(sel.tolist()).issubset(set(range(len(ds)))), "Selected indices are out of candidate range."

    assert math.isfinite(float(out.scoring_time_sec))
    assert math.isfinite(float(out.selection_time_sec))

    print("SAAL smoke tests passed.")


if __name__ == "__main__":
    main()

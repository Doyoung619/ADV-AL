import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from acquisition.logdet_adv_disp import (
    LogDetAdvDispStrategy,
    compute_adv_displacement_embeddings,
    greedy_logdet_selector,
)


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


def _assert_inverse_update_matches_direct(d: torch.Tensor, picked_local: np.ndarray, lam: float):
    eye = torch.eye(d.size(1), dtype=d.dtype)
    a_inv = eye / lam
    selected = []
    for idx in picked_local.tolist():
        u = d[idx]
        score_sm = float(torch.dot(u, a_inv @ u).item())
        assert math.isfinite(score_sm), "Greedy score is not finite."

        selected_tensor = d[selected] if len(selected) > 0 else torch.empty((0, d.size(1)), dtype=d.dtype)
        a_direct = lam * eye + selected_tensor.t() @ selected_tensor
        a_direct_inv = torch.linalg.inv(a_direct)
        score_direct = float(torch.dot(u, a_direct_inv @ u).item())
        assert np.isclose(score_sm, score_direct, atol=1e-5, rtol=1e-4), "Score mismatch before rank-1 update."

        denom = 1.0 + score_sm
        assert denom > 0.0, "Sherman-Morrison denominator must be positive."
        v = a_inv @ u
        a_inv = a_inv - torch.outer(v, v) / denom
        a_inv = 0.5 * (a_inv + a_inv.t())

        selected.append(idx)
        selected_tensor = d[selected]
        a_after = lam * eye + selected_tensor.t() @ selected_tensor
        a_after_inv = torch.linalg.inv(a_after)
        assert torch.allclose(a_inv, a_after_inv, atol=1e-5, rtol=1e-4), "Rank-1 inverse update mismatch."


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    n = 12
    num_classes = 5

    # Keep images in [0,1] so clamp-to-valid-range is well-defined with mean=0/std=1.
    x = torch.rand(n, 3, 8, 8, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    ).to(device)
    model.eval()

    # Test 1: FGSM displacement generation shape + finiteness.
    disp_fgsm = compute_adv_displacement_embeddings(
        model=model,
        unlabeled_loader=loader,
        attack_type="fgsm",
        attack_norm="linf",
        epsilon=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=device,
    )
    assert disp_fgsm.shape == (n, num_classes), f"FGSM displacement shape mismatch: {tuple(disp_fgsm.shape)}"
    assert torch.isfinite(disp_fgsm).all(), "FGSM displacement contains non-finite values."

    # Test 2: PGD displacement generation shape + finiteness.
    disp_pgd = compute_adv_displacement_embeddings(
        model=model,
        unlabeled_loader=loader,
        attack_type="pgd",
        attack_norm="linf",
        epsilon=1.0 / 255.0,
        pgd_steps=5,
        pgd_step_size=None,  # auto epsilon / max(steps/2, 1)
        pgd_random_start=True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=device,
    )
    assert disp_pgd.shape == (n, num_classes), f"PGD displacement shape mismatch: {tuple(disp_pgd.shape)}"
    assert torch.isfinite(disp_pgd).all(), "PGD displacement contains non-finite values."

    # Test 3: Greedy selection returns k unique indices and finite scores.
    lam = 1e-3
    picked_local, debug = greedy_logdet_selector(
        displacements=disp_pgd,
        query_size=4,
        lambda_reg=lam,
        score_chunk_size=3,
        jitter=1e-8,
    )
    assert len(picked_local) == 4, f"Expected 4 selections, got {len(picked_local)}"
    assert len(np.unique(picked_local)) == 4, "Greedy selection contains duplicates."
    assert all(math.isfinite(float(s)) for s in debug["selected_scores"]), "Greedy selected scores are not finite."

    # Test 4: Sherman-Morrison rank-1 updates match direct inverse.
    _assert_inverse_update_matches_direct(d=disp_pgd.to(dtype=torch.float64), picked_local=picked_local, lam=lam)

    # Test 5: End-to-end strategy API contract.
    cfg = SimpleNamespace(
        logdet_adv_disp_attack="fgsm",
        logdet_adv_disp_attack_norm="linf",
        logdet_adv_disp_epsilon=1.0 / 255.0,
        logdet_adv_disp_lambda=1e-3,
        logdet_adv_disp_pgd_steps=5,
        logdet_adv_disp_pgd_step_size=None,
        logdet_adv_disp_pgd_random_start=True,
        logdet_adv_disp_score_chunk_size=8,
        logdet_adv_disp_jitter=1e-8,
        cifar10_mean=(0.0, 0.0, 0.0),
        cifar10_std=(1.0, 1.0, 1.0),
        pool_batch_size=4,
        debug_save_adv_scores=False,
    )
    strategy = LogDetAdvDispStrategy(cfg)
    out = strategy.select(
        model=model,
        unlabeled_loader=loader,
        labeled_loader=None,
        unlabeled_indices=np.arange(n, dtype=np.int64),
        budget=4,
        device=device,
        progress_logger=None,
    )
    assert len(out.selected_indices) == 4, f"Strategy expected 4 selections, got {len(out.selected_indices)}"
    assert len(np.unique(out.selected_indices)) == 4, "Strategy selected duplicate indices."
    assert set(out.selected_indices.tolist()).issubset(set(range(n))), "Strategy selected out-of-range index."
    assert math.isfinite(float(out.scoring_time_sec)), "Non-finite scoring time."
    assert math.isfinite(float(out.selection_time_sec)), "Non-finite selection time."
    assert out.scores is not None and out.scores.shape == (n,), "Expected first-step scores of shape [N]."

    print("LogDet adversarial displacement smoke tests passed.")


if __name__ == "__main__":
    main()

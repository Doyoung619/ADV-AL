import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from acquisition import build_acquisition_strategy
from acquisition.logdet_adv_disp import compute_adv_semantic_displacement_embeddings, forward_with_features
from models import build_model


class TinyIndexedDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], 0, int(idx)


def _make_cfg(acquisition_method: str) -> SimpleNamespace:
    return SimpleNamespace(
        logdet_adv_disp_attack="fgsm",
        logdet_adv_disp_attack_norm="linf",
        logdet_adv_disp_epsilon=1.0 / 255.0,
        logdet_adv_disp_lambda=1e-3,
        logdet_adv_disp_pgd_steps=5,
        logdet_adv_disp_pgd_step_size=None,
        logdet_adv_disp_pgd_random_start=True,
        logdet_adv_disp_score_chunk_size=16,
        logdet_adv_disp_jitter=1e-8,
        logdet_adv_disp_percentile=0.0,
        logdet_adv_disp_swap_max_rounds=3,
        logdet_adv_disp_swap_top_unselected=0,
        logdet_adv_disp_swap_top_selected=0,
        logdet_adv_disp_swap_improvement_tol=1e-8,
        logdet_adv_disp_swap_downdate_tol=1e-6,
        logdet_adv_disp_swap_jitter=1e-8,
        cifar10_mean=(0.0, 0.0, 0.0),
        cifar10_std=(1.0, 1.0, 1.0),
        pool_batch_size=4,
        debug_save_adv_scores=False,
        acquisition_method=acquisition_method,
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    # Test 1: forward_with_features shape checks for small_cnn and resnet18.
    x_feat = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    for model_name in ("small_cnn", "resnet18"):
        model = build_model(model_name=model_name, num_classes=10, dropout_p=0.2).to(device)
        model.eval()
        features, logits = forward_with_features(model, x_feat, require_features=True)
        assert features is not None, f"{model_name}: features should not be None"
        assert features.shape[0] == x_feat.shape[0], f"{model_name}: feature batch mismatch"
        assert logits.shape == (x_feat.shape[0], 10), f"{model_name}: logits shape mismatch {tuple(logits.shape)}"
        assert features.ndim == 2, f"{model_name}: expected rank-2 features"
        assert torch.isfinite(features).all() and torch.isfinite(logits).all(), f"{model_name}: non-finite outputs"

    # Build a tiny unlabeled loader.
    n = 8
    x = torch.rand(n, 3, 32, 32, dtype=torch.float32)
    ds = TinyIndexedDataset(x)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = build_model(model_name="small_cnn", num_classes=10, dropout_p=0.2).to(device)
    model.eval()

    # Test 2: label-aware feature/logit adversarial displacement shapes and finiteness.
    d_feat, clean_logits_feat = compute_adv_semantic_displacement_embeddings(
        model=model,
        unlabeled_loader=loader,
        embedding_space="features",
        attack_type="fgsm",
        attack_norm="linf",
        epsilon=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=device,
        return_clean_logits=True,
    )
    d_logit, clean_logits_logit = compute_adv_semantic_displacement_embeddings(
        model=model,
        unlabeled_loader=loader,
        embedding_space="logits",
        attack_type="fgsm",
        attack_norm="linf",
        epsilon=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=device,
        return_clean_logits=True,
    )
    assert d_feat.shape[0] == n and d_feat.shape[1] == int(model.feature_dim), f"Feature delta shape mismatch: {tuple(d_feat.shape)}"
    assert d_logit.shape == (n, 10), f"Logit delta shape mismatch: {tuple(d_logit.shape)}"
    assert clean_logits_feat.shape == (n, 10), "Clean logits shape mismatch for feature-space run"
    assert clean_logits_logit.shape == (n, 10), "Clean logits shape mismatch for logit-space run"
    assert torch.isfinite(d_feat).all() and torch.isfinite(d_logit).all(), "Non-finite semantic displacements"

    # Test 3: method registry + percentile aliases and end-to-end selection smoke check.
    method_names = [
        "logdet_adv_feat_swap",
        "logdet_adv_feat_swap_p10",
        "logdet_adv_feat_swap_p20",
        "logdet_adv_logit_swap",
        "logdet_adv_logit_swap_p10",
        "logdet_adv_logit_swap_p20",
    ]

    unlabeled_indices = np.arange(n, dtype=np.int64)
    for method_name in method_names:
        cfg = _make_cfg(acquisition_method=method_name)
        strategy = build_acquisition_strategy(method_name, cfg)

        if method_name.endswith("_p10"):
            assert abs(float(cfg.logdet_adv_disp_percentile) - 0.1) < 1e-12, "p10 percentile parse failed"
        if method_name.endswith("_p20"):
            assert abs(float(cfg.logdet_adv_disp_percentile) - 0.2) < 1e-12, "p20 percentile parse failed"

        out = strategy.select(
            model=model,
            unlabeled_loader=loader,
            labeled_loader=None,
            unlabeled_indices=unlabeled_indices,
            budget=4,
            device=device,
            progress_logger=None,
        )
        assert len(out.selected_indices) == 4, f"{method_name}: expected 4 selections"
        assert len(np.unique(out.selected_indices)) == 4, f"{method_name}: duplicate selections"
        assert set(out.selected_indices.tolist()).issubset(set(unlabeled_indices.tolist())), f"{method_name}: out-of-range index"
        assert out.scores is not None and out.scores.shape == (n,), f"{method_name}: invalid score shape"
        assert np.isfinite(out.scores).all(), f"{method_name}: non-finite scores"

        obj_before = float(out.extras["swap_initial_logdet_objective"])
        obj_after = float(out.extras["swap_final_logdet_objective"])
        assert math.isfinite(obj_before) and math.isfinite(obj_after), f"{method_name}: non-finite objectives"
        assert obj_after + 1e-6 >= obj_before, f"{method_name}: swap objective decreased"

    print("Label-aware semantic logdet smoke tests passed.")


if __name__ == "__main__":
    main()

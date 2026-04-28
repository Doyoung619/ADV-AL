from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _channel_tensor(values: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def _scaled_linf_eps(epsilon: float, std: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    std_t = _channel_tensor(std, device=device, dtype=dtype)
    return torch.full((1, len(std), 1, 1), float(epsilon), device=device, dtype=dtype) / std_t


def _clamp_to_valid_range(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = _channel_tensor(mean, x.device, x.dtype)
    std_t = _channel_tensor(std, x.device, x.dtype)
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t
    return torch.max(torch.min(x, upper), lower)


def fgsm_attack(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    adv = x + eps_t * grad.sign()
    adv = _clamp_to_valid_range(adv, mean, std).detach()
    return adv


def pgd_attack(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    random_start: bool = True,
) -> torch.Tensor:
    model.eval()
    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    alpha_t = _scaled_linf_eps(alpha, std, images.device, images.dtype)

    x0 = images.detach()
    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(x0)

    x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    return x_adv.detach()


def fgsm_logit_mismatch_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        clean_logits = model(x0)

    x_adv = x0.clone().detach().requires_grad_(True)
    adv_logits = model(x_adv)
    mismatch_sq = (adv_logits - clean_logits).pow(2).sum(dim=1).mean()
    grad = torch.autograd.grad(mismatch_sq, x_adv, only_inputs=True)[0]

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    x_adv = x0 + eps_t * grad.sign()
    x_adv = _clamp_to_valid_range(x_adv, mean, std).detach()
    return x_adv


def pgd_logit_mismatch_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    random_start: bool = True,
) -> torch.Tensor:
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        clean_logits = model(x0)

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    alpha_t = _scaled_linf_eps(alpha, std, images.device, images.dtype)

    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(x0)
    x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        adv_logits = model(x_adv)
        mismatch_sq = (adv_logits - clean_logits).pow(2).sum(dim=1).mean()
        grad = torch.autograd.grad(mismatch_sq, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    return x_adv.detach()


def fgsm_predictive_ce_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """
    FGSM using pseudo-label CE objective:
      y_hat = argmax model(x)
      delta maximizes CE(model(x+delta), y_hat)
    """
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        pseudo_labels = model(x0).argmax(dim=1)

    x_adv = x0.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, pseudo_labels)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    x_adv = x0 + eps_t * grad.sign()
    x_adv = _clamp_to_valid_range(x_adv, mean, std).detach()
    return x_adv


def pgd_predictive_ce_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD using pseudo-label CE objective:
      y_hat = argmax model(x)
      delta maximizes CE(model(x+delta), y_hat)
    """
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        pseudo_labels = model(x0).argmax(dim=1)

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    alpha_t = _scaled_linf_eps(alpha, std, images.device, images.dtype)

    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(x0)
    x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, pseudo_labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    return x_adv.detach()


def _logit_gap(
    logits: torch.Tensor,
    fixed_top2_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute top-1/top-2 logit gap per sample.
    - logits: [B, C]
    - fixed_top2_idx: [B, 2] clean top-1/top-2 class indices to reuse.
    Returns: [B]
    """
    if fixed_top2_idx is None:
        top2_vals = torch.topk(logits, k=2, dim=1).values
        return top2_vals[:, 0] - top2_vals[:, 1]

    z1 = logits.gather(1, fixed_top2_idx[:, 0:1]).squeeze(1)
    z2 = logits.gather(1, fixed_top2_idx[:, 1:2]).squeeze(1)
    return z1 - z2


def pgd_gap_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    use_fixed_clean_classes: bool = True,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD attack maximizing squared gap change objective:
      max ||delta||_inf<=eps  | g(x+delta) - g(x) |^2
    where g(v) = z_(1)(v) - z_(2)(v).

    By default, top-1/top-2 classes are fixed from clean logits for stability.
    """
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        clean_logits = model(x0)
        clean_top2_idx = torch.topk(clean_logits, k=2, dim=1).indices
        clean_gap = _logit_gap(
            clean_logits,
            fixed_top2_idx=clean_top2_idx if use_fixed_clean_classes else None,
        )

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    alpha_t = _scaled_linf_eps(alpha, std, images.device, images.dtype)

    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(x0)
    x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    fixed_idx = clean_top2_idx if use_fixed_clean_classes else None
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        adv_gap = _logit_gap(logits, fixed_top2_idx=fixed_idx)
        loss = (adv_gap - clean_gap).pow(2).mean()
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    return x_adv.detach()


def _last_layer_grad_coeff(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> torch.Tensor:
    """
    For CE loss with fixed pseudo-label y_hat, gradient wrt last-layer weight W is:
      u_W = (p - one_hot(y_hat)) ⊗ h
    This helper returns (p - one_hot(y_hat)) part, shape [B, C].
    """
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(pseudo_labels, num_classes=probs.size(1)).to(dtype=probs.dtype)
    return probs - one_hot


def _grad_disp_sq_from_coeff_feat(
    coeff_a: torch.Tensor,
    feat_a: torch.Tensor,
    coeff_b: torch.Tensor,
    feat_b: torch.Tensor,
) -> torch.Tensor:
    """
    Exact per-sample ||(coeff_a ⊗ feat_a) - (coeff_b ⊗ feat_b)||_F^2 without materializing [B,C,D].
    Returns shape [B].
    """
    norm_a = coeff_a.pow(2).sum(dim=1) * feat_a.pow(2).sum(dim=1)
    norm_b = coeff_b.pow(2).sum(dim=1) * feat_b.pow(2).sum(dim=1)
    inner = (coeff_a * coeff_b).sum(dim=1) * (feat_a * feat_b).sum(dim=1)
    return (norm_a + norm_b - 2.0 * inner).clamp_min(0.0)


def fgsm_grad_disp_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """
    FGSM maximizing gradient displacement score:
      ||u_W(x+delta) - u_W(x)||_F^2
    with pseudo-label fixed from clean x.
    """
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        clean_logits, clean_features = model(x0, return_features=True)
        pseudo_labels = clean_logits.argmax(dim=1)
        clean_coeff = _last_layer_grad_coeff(clean_logits, pseudo_labels)

    x_adv = x0.clone().detach().requires_grad_(True)
    adv_logits, adv_features = model(x_adv, return_features=True)
    adv_coeff = _last_layer_grad_coeff(adv_logits, pseudo_labels)
    disp_sq = _grad_disp_sq_from_coeff_feat(adv_coeff, adv_features, clean_coeff, clean_features).mean()
    grad = torch.autograd.grad(disp_sq, x_adv, only_inputs=True)[0]

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    x_adv = x0 + eps_t * grad.sign()
    x_adv = _clamp_to_valid_range(x_adv, mean, std).detach()
    return x_adv


def pgd_grad_disp_attack(
    model,
    images: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mean: Sequence[float],
    std: Sequence[float],
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD maximizing gradient displacement score:
      max ||delta||_inf<=eps ||u_W(x+delta) - u_W(x)||_F^2
    where u_W is last-layer CE gradient with fixed clean pseudo-label y_hat.
    """
    model.eval()
    x0 = images.detach()
    with torch.no_grad():
        clean_logits, clean_features = model(x0, return_features=True)
        # Keep pseudo-label fixed from clean x to avoid objective discontinuity
        # caused by class switching during optimization.
        pseudo_labels = clean_logits.argmax(dim=1)
        clean_coeff = _last_layer_grad_coeff(clean_logits, pseudo_labels)

    eps_t = _scaled_linf_eps(epsilon, std, images.device, images.dtype)
    alpha_t = _scaled_linf_eps(alpha, std, images.device, images.dtype)

    if random_start:
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps_t
    else:
        delta = torch.zeros_like(x0)
    x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        adv_logits, adv_features = model(x_adv, return_features=True)
        adv_coeff = _last_layer_grad_coeff(adv_logits, pseudo_labels)
        disp_sq = _grad_disp_sq_from_coeff_feat(adv_coeff, adv_features, clean_coeff, clean_features).mean()
        grad = torch.autograd.grad(disp_sq, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha_t * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps_t, max=eps_t)
        x_adv = _clamp_to_valid_range(x0 + delta, mean, std)

    return x_adv.detach()


@torch.no_grad()
def grad_disp_score_from_logits_features(
    clean_logits: torch.Tensor,
    clean_features: torch.Tensor,
    adv_logits: torch.Tensor,
    adv_features: torch.Tensor,
    pseudo_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-sample score:
      ||u_W(x_adv) - u_W(x)||_F^2
    with fixed pseudo-labels from clean logits by default.
    """
    if pseudo_labels is None:
        pseudo_labels = clean_logits.argmax(dim=1)
    clean_coeff = _last_layer_grad_coeff(clean_logits, pseudo_labels)
    adv_coeff = _last_layer_grad_coeff(adv_logits, pseudo_labels)
    return _grad_disp_sq_from_coeff_feat(adv_coeff, adv_features, clean_coeff, clean_features)

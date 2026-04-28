from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features_logits_probs(model, loader, device: torch.device):
    model.eval()
    all_features = []
    all_logits = []
    all_probs = []
    all_indices = []

    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        probs = F.softmax(logits, dim=1)

        all_features.append(features.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_indices.append(indices.clone())

    features = torch.cat(all_features, dim=0)
    logits = torch.cat(all_logits, dim=0)
    probs = torch.cat(all_probs, dim=0)
    indices = torch.cat(all_indices, dim=0)
    return features, logits, probs, indices


@torch.no_grad()
def extract_features_probs(model, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, _, probs, indices = extract_features_logits_probs(model, loader, device)
    return features, probs, indices

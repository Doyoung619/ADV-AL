from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class IndexedSubset(Dataset):
    """Subset wrapper that always returns original dataset index."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_idx = int(self.indices[item])
        image, label = self.base_dataset[base_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, base_idx


def get_cifar10_transforms(mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def load_cifar10(data_dir: str, download_if_missing: bool = False):
    """
    Load CIFAR-10 from local storage by default.
    If not found and download_if_missing=True, download automatically.
    """
    try:
        train_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=None)
        test_base = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=None)
        return train_base, test_base
    except RuntimeError as e:
        if not download_if_missing:
            raise RuntimeError(
                f"CIFAR-10 not found in data_dir='{data_dir}'. "
                f"Place dataset there or run with --download-if-missing."
            ) from e
        train_base = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_base = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        return train_base, test_base


def build_loader(
    base_dataset: Dataset,
    indices: Sequence[int],
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    subset = IndexedSubset(base_dataset=base_dataset, indices=indices, transform=transform)
    loader_kwargs = dict(
        dataset=subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        # On Windows, worker process spawn overhead is large.
        # Keep workers alive across epochs to avoid per-epoch respawn cost.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def split_train_val(indices: Sequence[int], val_split: float, seed: int):
    idx = np.asarray(indices, dtype=np.int64).copy()
    if val_split <= 0.0:
        return idx, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_split))
    n_val = max(1, min(n_val, len(idx) - 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def full_train_indices(train_base: Dataset) -> np.ndarray:
    return np.arange(len(train_base), dtype=np.int64)


def full_test_indices(test_base: Dataset) -> np.ndarray:
    return np.arange(len(test_base), dtype=np.int64)

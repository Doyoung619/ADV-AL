import os
import urllib.request
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


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


class TinyImageNetValDataset(Dataset):
    """
    Tiny-ImageNet val loader for canonical layout:
    val/
      images/*.JPEG
      val_annotations.txt
    """

    def __init__(self, root: str, class_to_idx: Dict[str, int] = None):
        images_dir = os.path.join(root, "images")
        ann_path = os.path.join(root, "val_annotations.txt")
        if not os.path.isdir(images_dir):
            raise RuntimeError(f"Tiny-ImageNet val images dir not found: {images_dir}")
        if not os.path.exists(ann_path):
            raise RuntimeError(f"Tiny-ImageNet val annotations not found: {ann_path}")

        samples = []
        class_to_idx = {} if class_to_idx is None else dict(class_to_idx)
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    if class_to_idx:
                        raise RuntimeError(f"Tiny-ImageNet val class not present in train classes: {wnid}")
                    class_to_idx[wnid] = len(class_to_idx)
                label = class_to_idx[wnid]
                img_path = os.path.join(images_dir, img_name)
                samples.append((img_path, label))
        if not samples:
            raise RuntimeError(f"No Tiny-ImageNet val samples found under: {root}")

        self.samples = samples
        self.targets = [y for _, y in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return image, int(label)


DATASET_STATS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    # Standard CIFAR-10 normalization.
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    # Standard CIFAR-100 normalization.
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    # Tiny-ImageNet commonly uses ImageNet normalization.
    "tinyimagenet": {
        "mean": (0.4850, 0.4560, 0.4060),
        "std": (0.2290, 0.2240, 0.2250),
    },
    # CINIC-10 official channel-wise statistics.
    "cinic10": {
        "mean": (0.47889522, 0.47227842, 0.43047404),
        "std": (0.24205776, 0.23828046, 0.25874835),
    },
    # STL-10 commonly uses this channel-wise normalization.
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
    },
    # Common SVHN normalization (train split statistics).
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    # FashionMNIST converted to 3-channel: replicate grayscale statistics.
    "fashionmnist": {
        "mean": (0.2860, 0.2860, 0.2860),
        "std": (0.3530, 0.3530, 0.3530),
    },
}


def default_dataset_stats(dataset_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    key = dataset_name.lower()
    if key not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset for stats: {dataset_name}")
    stats = DATASET_STATS[key]
    return stats["mean"], stats["std"]


def get_dataset_transforms(
    dataset_name: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    key = dataset_name.lower()
    if key in {"cifar10", "cifar100", "cinic10"}:
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

    if key == "stl10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(96, padding=12),
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

    if key == "tinyimagenet":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
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

    if key == "svhn":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
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

    if key == "fashionmnist":
        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_transform, eval_transform

    raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")


def load_dataset(dataset_name: str, data_dir: str, download_if_missing: bool = False):
    key = dataset_name.lower()
    if key == "cifar10":
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

    if key == "cifar100":
        try:
            train_base = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=None)
            test_base = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=None)
            return train_base, test_base
        except RuntimeError as e:
            if not download_if_missing:
                raise RuntimeError(
                    f"CIFAR-100 not found in data_dir='{data_dir}'. "
                    f"Place dataset there or run with --download-if-missing."
                ) from e
            train_base = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
            test_base = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
            return train_base, test_base

    if key == "tinyimagenet":
        root = os.path.join(data_dir, "tiny-imagenet-200")
        train_root = os.path.join(root, "train")
        val_root = os.path.join(root, "val")
        if not os.path.isdir(train_root) or not os.path.isdir(val_root):
            if not download_if_missing:
                raise RuntimeError(
                    "Tiny-ImageNet not found. Expected directory layout: "
                    f"{root}/train and {root}/val. "
                    "Download/extract tiny-imagenet-200 manually or run with --download-if-missing."
                )
            os.makedirs(data_dir, exist_ok=True)
            zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
            if not os.path.exists(zip_path):
                print(f"[data] downloading Tiny-ImageNet from {TINY_IMAGENET_URL} to {zip_path}")
                urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
            print(f"[data] extracting Tiny-ImageNet to {data_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            if not os.path.isdir(train_root) or not os.path.isdir(val_root):
                raise RuntimeError(
                    "Tiny-ImageNet download/extract completed but expected layout was not found: "
                    f"{root}/train and {root}/val"
                )
        train_base = datasets.ImageFolder(root=train_root, transform=None)
        val_images_root = os.path.join(val_root, "images")
        # Support both canonical val layout and pre-reorganized val folders.
        if os.path.isdir(val_images_root) and os.path.exists(os.path.join(val_root, "val_annotations.txt")):
            test_base = TinyImageNetValDataset(root=val_root, class_to_idx=train_base.class_to_idx)
        else:
            test_base = datasets.ImageFolder(root=val_root, transform=None)
        return train_base, test_base

    if key == "cinic10":
        root = os.path.join(data_dir, "CINIC-10")
        train_root = os.path.join(root, "train")
        test_root = os.path.join(root, "test")
        if not os.path.isdir(train_root) or not os.path.isdir(test_root):
            raise RuntimeError(
                "CINIC-10 not found. Expected directory layout: "
                f"{root}/train and {root}/test. "
                "Download/extract CINIC-10 manually and set --data-dir accordingly."
            )
        train_base = datasets.ImageFolder(root=train_root, transform=None)
        test_base = datasets.ImageFolder(root=test_root, transform=None)
        return train_base, test_base

    if key == "stl10":
        try:
            train_base = datasets.STL10(root=data_dir, split="train", download=False, transform=None)
            test_base = datasets.STL10(root=data_dir, split="test", download=False, transform=None)
            return train_base, test_base
        except RuntimeError as e:
            if not download_if_missing:
                raise RuntimeError(
                    f"STL-10 not found in data_dir='{data_dir}'. "
                    f"Place dataset there or run with --download-if-missing."
                ) from e
            train_base = datasets.STL10(root=data_dir, split="train", download=True, transform=None)
            test_base = datasets.STL10(root=data_dir, split="test", download=True, transform=None)
            return train_base, test_base

    if key == "svhn":
        try:
            train_base = datasets.SVHN(root=data_dir, split="train", download=False, transform=None)
            test_base = datasets.SVHN(root=data_dir, split="test", download=False, transform=None)
            return train_base, test_base
        except RuntimeError as e:
            if not download_if_missing:
                raise RuntimeError(
                    f"SVHN not found in data_dir='{data_dir}'. "
                    f"Place dataset there or run with --download-if-missing."
                ) from e
            train_base = datasets.SVHN(root=data_dir, split="train", download=True, transform=None)
            test_base = datasets.SVHN(root=data_dir, split="test", download=True, transform=None)
            return train_base, test_base

    if key == "fashionmnist":
        try:
            train_base = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=None)
            test_base = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=None)
            return train_base, test_base
        except RuntimeError as e:
            if not download_if_missing:
                raise RuntimeError(
                    f"FashionMNIST not found in data_dir='{data_dir}'. "
                    f"Place dataset there or run with --download-if-missing."
                ) from e
            train_base = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=None)
            test_base = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=None)
            return train_base, test_base

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_cifar10_transforms(mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    # Backward-compatible alias.
    return get_dataset_transforms("cifar10", mean=mean, std=std)


def load_cifar10(data_dir: str, download_if_missing: bool = False):
    # Backward-compatible alias.
    return load_dataset("cifar10", data_dir=data_dir, download_if_missing=download_if_missing)


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

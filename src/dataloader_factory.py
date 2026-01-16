# data_setup.py — prepares datasets & dataloaders using augmentation.py + config_aug.py

from __future__ import annotations
import os
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

# Single source of truth + transform builder
from .config_aug_fix import AUGMENTATION_CONFIG
from .aug_albumentation_2 import build_transforms


# Flat folder dataset for inference/submission (no class subfolders)
class FlatImages(Dataset):
    def __init__(self, root: str, transform=None,
                 img_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        self.root = root
        self.transform = transform
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith(img_exts)]
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # return (tensor, dummy_label, filename)
        return img, -1, os.path.basename(p)


def prepare_data(
    train_dir: str,
    valid_or_test_dir: str,
    cfg: dict = AUGMENTATION_CONFIG,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    use_flat_test: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train & valid/test DataLoaders controlled by config_aug.py.
    - train_dir, valid_or_test_dir: classification folders with class subfolders
      (set use_flat_test=True if valid_or_test_dir is a flat folder of images)
    Returns: (train_loader, test_loader, class_names)
    """

    # Build transforms from the shared config
    train_transform, test_transform = build_transforms(cfg)

    # Datasets using those transforms
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    if use_flat_test:
        test_data = FlatImages(valid_or_test_dir, transform=test_transform)
        class_names = train_data.classes
    else:
        test_data = datasets.ImageFolder(valid_or_test_dir, transform=test_transform)
        class_names = train_data.classes

    # DataLoader knobs (from config, with optional overrides)
    data_cfg = cfg["data"]
    bs  = batch_size  if batch_size  is not None else data_cfg.get("batch_size", 32)
    nw  = num_workers if num_workers is not None else data_cfg.get("num_workers", 4)
    pin = pin_memory if pin_memory is not None else data_cfg.get("pin_memory", True)

    # Loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    return train_dataloader, test_dataloader, class_names


def create_submission_dataloader(
    images_dir: str,
    cfg: dict = AUGMENTATION_CONFIG,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Build a DataLoader for a folder of unlabeled images (no subfolders)."""
    _, test_transform = build_transforms(cfg)
    ds = FlatImages(images_dir, transform=test_transform)

    data_cfg = cfg["data"]
    bs  = batch_size  if batch_size  is not None else data_cfg.get("batch_size", 32)
    nw  = num_workers if num_workers is not None else data_cfg.get("num_workers", 4)
    pin = pin_memory if pin_memory is not None else data_cfg.get("pin_memory", True)

    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)


if __name__ == "__main__":
    # quick smoke test (adjust paths)
    train_dir = "data/train"
    val_dir   = "data/val"

    train_loader, val_loader, classes = prepare_data(
        train_dir, val_dir, cfg=AUGMENTATION_CONFIG, use_flat_test=False
    )
    xb, yb = next(iter(train_loader))[:2]
    print("Classes:", classes)
    print("Train batch:", xb.shape, yb.shape)

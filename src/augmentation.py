# augmentation.py — builds train/test tensor transforms from config_aug.py

from __future__ import annotations
from typing import Tuple, Sequence, Optional

from torchvision import transforms
try:
    from torchvision.transforms import TrivialAugmentWide, RandAugment
except Exception:
    TrivialAugmentWide = None
    RandAugment = None


def _to_hw(size: Optional[Sequence[int] | int]):
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError("image_size must be int or [H, W]")


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, test_transform) as torchvision transforms producing tensors."""
    data_cfg  = cfg["data"]
    train_cfg = cfg.get("train", {})
    test_cfg  = cfg.get("test",  {})

    image_size = data_cfg.get("image_size", 384)
    im_hw = _to_hw(image_size)
    mean = data_cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std  = data_cfg.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])

    # ---------------- train ----------------
    t = []
    if train_cfg.get("random_resized_crop", {}).get("enabled", False):
        s = train_cfg["random_resized_crop"].get("scale", [0.85, 1.0])
        r = train_cfg["random_resized_crop"].get("ratio", [0.75, 1.33])
        t.append(transforms.RandomResizedCrop(im_hw[0] if im_hw else image_size, scale=tuple(s), ratio=tuple(r)))
    elif train_cfg.get("resize", {}).get("enabled", False):
        size = train_cfg["resize"].get("size") or im_hw or (image_size, image_size)
        if isinstance(size, int): size = (size, size)
        t.append(transforms.Resize(size))

    if train_cfg.get("horizontal_flip", {}).get("enabled", False):
        t.append(transforms.RandomHorizontalFlip(p=train_cfg["horizontal_flip"].get("p", 0.5)))
    if train_cfg.get("vertical_flip", {}).get("enabled", False):
        t.append(transforms.RandomVerticalFlip(p=train_cfg["vertical_flip"].get("p", 0.5)))
    if train_cfg.get("rotation", {}).get("enabled", False):
        t.append(transforms.RandomRotation(degrees=train_cfg["rotation"].get("degrees", 15)))
    if train_cfg.get("affine", {}).get("enabled", False):
        af = train_cfg["affine"]
        t.append(transforms.RandomAffine(
            degrees=af.get("degrees", 0),
            translate=(af.get("translate", 0.0), af.get("translate", 0.0)),
            scale=None if "scale" not in af else tuple(af["scale"]),
            shear=af.get("shear", 0),
        ))
    if train_cfg.get("color_jitter", {}).get("enabled", False):
        cj = train_cfg["color_jitter"]
        t.append(transforms.ColorJitter(
            brightness=cj.get("brightness", 0.15),
            contrast=cj.get("contrast", 0.15),
            saturation=cj.get("saturation", 0.15),
            hue=cj.get("hue", 0.02),
        ))
    if train_cfg.get("trivial_augment_wide", {}).get("enabled", False) and TrivialAugmentWide:
        t.append(TrivialAugmentWide())
    if train_cfg.get("randaugment", {}).get("enabled", False) and RandAugment:
        ra = train_cfg["randaugment"]
        t.append(RandAugment(num_ops=ra.get("num_ops", 2), magnitude=ra.get("magnitude", 9)))

    t.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    train_transform = transforms.Compose(t)

    # ---------------- test/valid ----------------
    tt = []
    if test_cfg.get("resize", {}).get("enabled", True):
        size = test_cfg["resize"].get("size")
        size = _to_hw(size) if size is not None else (im_hw or (image_size, image_size))
        tt.append(transforms.Resize(size))
    if test_cfg.get("center_crop", {}).get("enabled", False):
        cc_size = test_cfg["center_crop"].get("size")
        if cc_size is None:
            cc_size = image_size if isinstance(image_size, int) else image_size[0]
        tt.append(transforms.CenterCrop(cc_size))
    tt.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transform = transforms.Compose(tt)

    return train_transform, test_transform

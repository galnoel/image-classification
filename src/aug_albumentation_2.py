# aug_albumentation.py — Albumentations-based transforms (pickle-safe for Windows DataLoader)
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any, List, Callable

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import inspect

# ---------------- utils ----------------

def _to_hw(size: Optional[Sequence[int] | int]):
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError("image_size must be int or [H, W]")

def _supports_size_kw(transform_cls):
    """Compat helper for different Albumentations versions (height/width vs size kw)."""
    try:
        sig = inspect.signature(transform_cls.__init__)
    except (TypeError, ValueError):
        return False
    return any(p.name == "size" for p in sig.parameters.values())

def _make_random_resized_crop(h: int, w: int, scale, ratio, p: float):
    if _supports_size_kw(A.RandomResizedCrop):
        return A.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio, p=p)
    return A.RandomResizedCrop(height=h, width=w, scale=scale, ratio=ratio, p=p)

def _make_resize(h: int, w: int, interpolation=cv2.INTER_AREA):
    if _supports_size_kw(A.Resize):
        return A.Resize(size=(h, w), interpolation=interpolation)
    return A.Resize(height=h, width=w, interpolation=interpolation)

def _make_center_crop(h: int, w: int):
    if _supports_size_kw(A.CenterCrop):
        return A.CenterCrop(size=(h, w))
    return A.CenterCrop(height=h, width=w)

# ---------------- custom transforms (optional extras) ----------------

from albumentations.core.transforms_interface import ImageOnlyTransform

class AHETransform(ImageOnlyTransform):
    def __init__(self, tile_grid_size=(8,8), clip_limit=4.0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.tile_grid_size = tuple(tile_grid_size)
        self.clip_limit = float(clip_limit)

    def apply(self, img, **params):
        img = img.copy()
        if img.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            return clahe.apply(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

class DehazeSimple(ImageOnlyTransform):
    def __init__(self, omega=0.95, t0=0.1, radius=7, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.omega = float(omega)
        self.t0 = float(t0)
        self.radius = int(radius)

    def _dark_channel(self, img, radius):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
        min_per_channel = img.min(axis=2)
        dark = cv2.erode(min_per_channel, kernel)
        return dark

    def apply(self, img, **params):
        I = img.astype(np.float32) / 255.0
        dark = self._dark_channel(I, self.radius)
        flat = dark.reshape(-1)
        topk = max(1, int(flat.size * 0.001))
        idx = np.argpartition(flat, -topk)[-topk:]
        A_atm = I.reshape(-1, 3)[idx].mean(axis=0)
        norm_I = I / (A_atm.reshape(1,1,3) + 1e-6)
        t = 1.0 - self.omega * self._dark_channel(norm_I, self.radius)
        t = np.clip(t, self.t0, 1.0)
        J = (I - A_atm.reshape(1,1,3)) / t[..., None] + A_atm.reshape(1,1,3)
        J = np.clip(J * 255.0, 0, 255).astype(np.uint8)
        return J

# ---------------- pickle-safe wrapper ----------------

class ComposeCallable:
    """Top-level callable wrapper so DataLoader workers (spawn) can pickle it on Windows."""
    def __init__(self, compose: A.Compose):
        self.compose = compose

    def __call__(self, img):
        # Convert incoming image to RGB numpy
        if hasattr(img, "mode"):  # PIL.Image
            img = np.array(img.convert("RGB"))
        else:
            img = np.asarray(img)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        out = self.compose(image=img)
        tensor: torch.Tensor = out["image"]
        return tensor

# ---------------- internal helpers ----------------

def _add_if_enabled(t_list: List[A.BasicTransform], name: str, cfg: Dict[str, Any]):
    if not cfg or not cfg.get("enabled", False):
        return
    p = float(cfg.get("p", 0.5))

    if name == "random_resized_crop":
        scale = tuple(cfg.get("scale", (0.85, 1.0)))
        ratio = tuple(cfg.get("ratio", (0.75, 1.33)))
        size = cfg.get("size")
        if size is None:
            raise ValueError("random_resized_crop requires 'size' in config (int or [H, W])")
        h, w = _to_hw(size)
        t_list.append(_make_random_resized_crop(h, w, scale=scale, ratio=ratio, p=1.0))
        return

    if name == "resize":
        size = cfg.get("size")
        if size is None:
            return
        h, w = _to_hw(size)
        t_list.append(_make_resize(h, w, interpolation=cv2.INTER_AREA))
        return

    if name == "horizontal_flip":
        t_list.append(A.HorizontalFlip(p=p))
    elif name == "vertical_flip":
        t_list.append(A.VerticalFlip(p=p))
    elif name == "rotation":
        t_list.append(A.Rotate(limit=cfg.get("degrees", 15), border_mode=cv2.BORDER_REFLECT_101, p=p))
    elif name == "affine":
        t_list.append(A.Affine(
            scale=tuple(cfg.get("scale", (1.0, 1.0))),
            translate_percent=(cfg.get("translate", 0.0), cfg.get("translate", 0.0)),
            rotate=cfg.get("degrees", 0),
            shear=cfg.get("shear", 0),
            fit_output=False, mode=cv2.BORDER_REFLECT_101, p=p
        ))
    elif name == "color_jitter":
        t_list.append(A.ColorJitter(
            brightness=cfg.get("brightness", 0.15),
            contrast=cfg.get("contrast", 0.15),
            saturation=cfg.get("saturation", 0.15),
            hue=cfg.get("hue", 0.02),
            p=p
        ))
    elif name == "brightness_contrast":
        t_list.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.15),
            contrast_limit=cfg.get("contrast_limit", 0.15),
            p=p
        ))
    elif name == "blur":
        t_list.append(A.Blur(blur_limit=cfg.get("blur_limit", 3), p=p))
    elif name == "gaussian_noise":
        t_list.append(A.GaussNoise(var_limit=tuple(cfg.get("var_limit", (10.0, 50.0))), p=p))
    elif name == "sharpen":
        t_list.append(A.Sharpen(alpha=tuple(cfg.get("alpha", (0.2, 0.5))), lightness=tuple(cfg.get("lightness", (0.5, 1.0))), p=p))
    elif name == "jpeg":
        t_list.append(A.ImageCompression(quality_lower=cfg.get("quality_lower", 80), quality_upper=cfg.get("quality_upper", 100), p=p))
    elif name == "grayscale":
        t_list.append(A.ToGray(p=p))
    elif name == "clahe":
        t_list.append(A.CLAHE(clip_limit=cfg.get("clip_limit", 4.0),
                              tile_grid_size=tuple(cfg.get("tile_grid_size", (8, 8))),
                              p=p))
    elif name == "ahe":
        t_list.append(AHETransform(tile_grid_size=tuple(cfg.get("tile_grid_size", (8, 8))),
                                   clip_limit=cfg.get("clip_limit", 8.0),
                                   p=p))
    elif name == "equalize":
        t_list.append(A.Equalize(mode=cfg.get("mode", "cv"),
                                 by_channels=cfg.get("by_channels", True),
                                 p=p))
    elif name == "denoise":
        t_list.append(A.Denoise(h=cfg.get("h", 10),
                                hColor=cfg.get("hColor", 10),
                                templateWindowSize=cfg.get("templateWindowSize", 7),
                                searchWindowSize=cfg.get("searchWindowSize", 21),
                                p=p))
    elif name == "defog":
        t_list.append(DehazeSimple(omega=float(cfg.get("omega", 0.95)),
                                   t0=float(cfg.get("t0", 0.1)),
                                   radius=int(cfg.get("radius", 7)),
                                   p=p))

# ---------------- public builder ----------------

def build_transforms(cfg, build_train: bool = True, build_test: bool = True):
    """Return (train_transform, test_transform) callables producing normalized tensors.

    This mirrors the signature/behavior of the torchvision-based module so it can be
    swapped without touching the rest of the pipeline.
    """
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    test_cfg  = cfg.get("test",  {})

    image_size = data_cfg.get("image_size", 384)
    im_hw = _to_hw(image_size)
    mean = data_cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std  = data_cfg.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])

    train_transform = None
    test_transform  = None

    # ---------- TRAIN ----------
    if build_train:
        t_list: List[A.BasicTransform] = []

        rr_cfg = train_cfg.get("random_resized_crop", {})
        if rr_cfg.get("enabled", False):
            size = rr_cfg.get("size") or im_hw or (image_size, image_size)
            rr_cfg_local = dict(rr_cfg)
            rr_cfg_local["size"] = size
            _add_if_enabled(t_list, "random_resized_crop", rr_cfg_local)
        else:
            rz_cfg = train_cfg.get("resize", {})
            if rz_cfg.get("enabled", False):
                size = rz_cfg.get("size") or im_hw or (image_size, image_size)
                _add_if_enabled(t_list, "resize", {"enabled": True, "size": size})

        # Geometric
        if train_cfg.get("horizontal_flip", {}).get("enabled", False):
            _add_if_enabled(t_list, "horizontal_flip", train_cfg["horizontal_flip"])
        if train_cfg.get("vertical_flip", {}).get("enabled", False):
            _add_if_enabled(t_list, "vertical_flip", train_cfg["vertical_flip"])
        if train_cfg.get("rotation", {}).get("enabled", False):
            _add_if_enabled(t_list, "rotation", train_cfg["rotation"])
        if train_cfg.get("affine", {}).get("enabled", False):
            _add_if_enabled(t_list, "affine", train_cfg["affine"])

        # Photometric
        if train_cfg.get("color_jitter", {}).get("enabled", False):
            _add_if_enabled(t_list, "color_jitter", train_cfg["color_jitter"])
        if train_cfg.get("brightness_contrast", {}).get("enabled", False):
            _add_if_enabled(t_list, "brightness_contrast", train_cfg["brightness_contrast"])
        if train_cfg.get("blur", {}).get("enabled", False):
            _add_if_enabled(t_list, "blur", train_cfg["blur"])
        if train_cfg.get("gaussian_noise", {}).get("enabled", False):
            _add_if_enabled(t_list, "gaussian_noise", train_cfg["gaussian_noise"])
        if train_cfg.get("sharpen", {}).get("enabled", False):
            _add_if_enabled(t_list, "sharpen", train_cfg["sharpen"])
        if train_cfg.get("jpeg", {}).get("enabled", False):
            _add_if_enabled(t_list, "jpeg", train_cfg["jpeg"])
        if train_cfg.get("grayscale", {}).get("enabled", False):
            _add_if_enabled(t_list, "grayscale", train_cfg["grayscale"])
        if train_cfg.get("clahe", {}).get("enabled", False):
            _add_if_enabled(t_list, "clahe", train_cfg["clahe"])
        if train_cfg.get("ahe", {}).get("enabled", False):
            _add_if_enabled(t_list, "ahe", train_cfg["ahe"])
        if train_cfg.get("equalize", {}).get("enabled", False):
            _add_if_enabled(t_list, "equalize", train_cfg["equalize"])
        if train_cfg.get("denoise", {}).get("enabled", False):
            _add_if_enabled(t_list, "denoise", train_cfg["denoise"])
        if train_cfg.get("defog", {}).get("enabled", False):
            _add_if_enabled(t_list, "defog", train_cfg["defog"])

        # Normalize + to tensor at the end
        t_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        train_transform = ComposeCallable(A.Compose(t_list))

    # ---------- TEST/VALID ----------
    if build_test:
        tt: List[A.BasicTransform] = []
        rz_cfg_t = test_cfg.get("resize", {"enabled": True, "size": im_hw or (image_size, image_size)})
        if rz_cfg_t.get("enabled", True):
            size = rz_cfg_t.get("size")
            size = _to_hw(size) if size is not None else (im_hw or (image_size, image_size))
            tt.append(_make_resize(size[0], size[1], interpolation=cv2.INTER_AREA))
        if test_cfg.get("center_crop", {}).get("enabled", False):
            cc_size = test_cfg["center_crop"].get("size")
            if cc_size is None:
                cc_size = image_size if isinstance(image_size, int) else image_size[0]
            cc_size = int(cc_size)
            tt.append(_make_center_crop(cc_size, cc_size))
        tt.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        test_transform = ComposeCallable(A.Compose(tt))

    return train_transform, test_transform

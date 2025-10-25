
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augmentation.py — minimal, config-driven augmentation

Files:
- augmentation.py (this file)
- config_aug.py     # provides AUGMENTATION_CONFIG dict

Usage:
  python augmentation.py

What it does:
- Reads AUGMENTATION_CONFIG from config_aug.py
- Builds Albumentations pipeline (rotate ON by default unless disabled)
- Batch-augments images from a manifest, writes augmented images and aug_manifest.csv

No CLI flags. All knobs live in config_aug.py.
"""
from __future__ import annotations
import os, csv, sys
from typing import Dict, Any, List, Tuple, Optional

import cv2
import albumentations as A

try:
    from config_aug import AUGMENTATION_CONFIG as CFG
except Exception as e:
    print("[FATAL] Cannot import AUGMENTATION_CONFIG from config_aug.py", file=sys.stderr)
    raise


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_abs(p: str) -> bool:
    return os.path.isabs(p)


def _ensure_hw(size):
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError("image_size must be int or (H, W)")


def _add_if_enabled(t_list, name: str, cfg: Dict[str, Any]):
    if not cfg or not cfg.get("enabled", False):
        return
    p = float(cfg.get("p", 0.5))
    if name == "rotate":
        limit = cfg.get("limit", 15)
        t_list.append(A.Rotate(limit=limit, border_mode=cv2.BORDER_REFLECT_101, p=p))
    elif name == "flip":
        if cfg.get("horizontal", True):
            t_list.append(A.HorizontalFlip(p=p))
        if cfg.get("vertical", False):
            t_list.append(A.VerticalFlip(p=p))
    elif name == "affine":
        t_list.append(A.Affine(
            scale=tuple(cfg.get("scale", (0.9, 1.1))),
            translate_percent=tuple(cfg.get("translate_percent", (0.0, 0.1))),
            rotate=cfg.get("rotate", 0),
            shear=tuple(cfg.get("shear", (-10, 10))),
            fit_output=False, mode=cv2.BORDER_REFLECT_101, p=p
        ))
    elif name == "brightness_contrast":
        t_list.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.15),
            contrast_limit=cfg.get("contrast_limit", 0.15), p=p))
    elif name == "color_jitter":
        t_list.append(A.ColorJitter(
            brightness=cfg.get("brightness", 0.1),
            contrast=cfg.get("contrast", 0.1),
            saturation=cfg.get("saturation", 0.1),
            hue=cfg.get("hue", 0.02), p=p))
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


def build_augmenter(conf: Dict[str, Any]) -> A.Compose:
    A.set_seed(int(conf.get("seed", 42)))
    transforms = conf.get("transforms", {})
    # rotate default ON if missing
    if "rotate" not in transforms:
        transforms["rotate"] = {"enabled": True, "limit": 15, "p": 0.5}

    t_list: List[A.BasicTransform] = []
    order = ["rotate","flip","affine","brightness_contrast","color_jitter","blur","gaussian_noise","sharpen","jpeg","grayscale"]
    for k in order:
        if k in transforms:
            _add_if_enabled(t_list, k, transforms[k])
    for k,v in transforms.items():
        if k not in order:
            _add_if_enabled(t_list, k, v)

    size = _ensure_hw(conf.get("image_size"))
    if size is not None:
        t_list.append(A.Resize(height=size[0], width=size[1], interpolation=cv2.INTER_AREA))
    return A.Compose(t_list)


def _read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    io = CFG["io"]
    manifest   = io["manifest"]
    images_root= io["images_root"]
    out_dir    = io["out_dir"]
    path_col   = io.get("path_col","filepath")
    label_col  = io.get("label_col")
    extra_cols = io.get("extra_cols", [])
    copies     = int(io.get("copies", 1))
    ext        = io.get("ext",".jpg")

    _ensure_dir(out_dir)
    rows = _read_rows(manifest)
    if not rows:
        print("[INFO] Empty manifest; nothing to do."); return

    aug = build_augmenter(CFG.get("augmentations", {}))

    out_manifest = os.path.join(os.path.dirname(out_dir.rstrip("/")), "aug_manifest.csv")
    fields = [path_col] + ([label_col] if label_col else []) + [c for c in extra_cols if c != label_col]
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        wrote = 0
        for r in rows:
            rel = r[path_col]; src = rel if _is_abs(rel) else os.path.join(images_root, rel)
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] skip unreadable: {src}", file=sys.stderr); continue
            base = os.path.splitext(os.path.basename(rel))[0]
            for k in range(copies):
                out = aug(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                name = f"{base}__aug{k:02d}{ext}"
                dst = os.path.join(out_dir, name)
                cv2.imwrite(dst, out_bgr)
                row_o = {path_col: os.path.relpath(dst, out_dir)}
                if label_col: row_o[label_col] = r.get(label_col,"")
                for c in extra_cols: row_o[c] = r.get(c,"")
                writer.writerow(row_o); wrote += 1
    print(f"[OK] Augmented images written; manifest: {out_manifest} ; out_dir: {out_dir}")


if __name__ == "__main__":
    main()

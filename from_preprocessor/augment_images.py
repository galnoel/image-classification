#!/usr/bin/env python3
"""
augment_images.py
Offline augmentation runner extracted from the notebook "imageprocessing-pipeline-v1-cat-dog-fixed.ipynb".

Usage:
  python augment_images.py --config augment_config.yaml

This script:
- Reads a manifest CSV with columns: filepath,label,split
- Applies (optional) deterministic preprocessing: manipulation + enhancement
- Applies stochastic data augmentation (Albumentations if available; otherwise a torchvision fallback)
- Saves augmented images into an output directory (train split by default)
- Emits an updated manifest CSV that includes the new augmented samples

Dependencies:
  pip install albumentations>=1.3.0 opencv-python-headless>=4.8.0 pillow pandas pyyaml tqdm torchvision torch matplotlib
"""

import os, sys, math, random, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- Optional backends ---
_have_albu = False
try:
    import albumentations as A
    _have_albu = True
except Exception:
    _have_albu = False

try:
    import cv2
except Exception:
    cv2 = None

# torchvision fallback for transforms if Albumentations unavailable
try:
    import torch
    import torchvision.transforms as T
except Exception:
    torch = None
    T = None

# -------------------------
# Utilities
# -------------------------

def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_image_cv2_rgb(fp: str):
    if cv2 is None:
        # fallback to PIL, then convert to NumPy RGB
        with Image.open(fp) as im:
            return np.array(im.convert('RGB'))
    cv2.setNumThreads(0)
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        with Image.open(fp) as im:
            img = np.array(im.convert('RGB'))[:, :, ::-1]  # RGB->BGR fallback read
        img = img[:, :, ::-1]  # back to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_cv2_rgb(fp: str, img_rgb: np.ndarray, quality: int = 95):
    # fp extension decides format
    ext = Path(fp).suffix.lower()
    if cv2 is not None and ext in {'.jpg', '.jpeg', '.png', '.webp'}:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if ext in {'.jpg', '.jpeg'}:
            cv2.imwrite(fp, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif ext == '.png':
            cv2.imwrite(fp, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:
            cv2.imwrite(fp, bgr)
    else:
        Image.fromarray(img_rgb).save(fp)

# -------------------------
# Deterministic preprocessing blocks (from notebook)
# -------------------------

def _letterbox_rgb(img: np.ndarray, target_hw=(640,640), pad_value=(114,114,114)):
    """Pad to square while keeping aspect ratio."""
    h, w = img.shape[:2]
    th, tw = target_hw
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if cv2 is None:
        # PIL fallback
        im = Image.fromarray(img)
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (tw, th), tuple(pad_value))
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        canvas.paste(im, (ox, oy))
        return np.array(canvas)
    else:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), pad_value, dtype=resized.dtype)
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        return canvas

def build_manipulation(cfg: dict):
    mode = cfg.get("size", {}).get("mode", "letterbox")
    lb = cfg.get("size", {}).get("letterbox", {"target_hw":[640,640], "pad_value":[114,114,114]})
    resize = cfg.get("size", {}).get("resize", {"size":[640,640]})
    norm = cfg.get("to_float", {"enabled": False, "max_value":255.0})

    def _manip(img: np.ndarray):
        out = img
        if mode == "letterbox":
            out = _letterbox_rgb(out, tuple(lb.get("target_hw", [640,640])), tuple(lb.get("pad_value", [114,114,114])))
        elif mode == "resize":
            size = tuple(resize.get("size", [640,640]))
            if cv2 is None:
                out = np.array(Image.fromarray(out).resize(size, Image.BILINEAR))
            else:
                out = cv2.resize(out, size, interpolation=cv2.INTER_LINEAR)
        # to_float at the end (optional)
        if norm.get("enabled", False):
            mv = float(norm.get("max_value", 255.0))
            out = out.astype(np.float32) / mv
        return out
    return _manip

def build_enhancement(cfg: dict):
    """Builds a deterministic Albumentations Compose from config.
       If Albumentations isn't available, falls back to identity."""
    if not _have_albu:
        def _id(img): return img
        return _id

    steps = []

    # Denoise
    den = cfg.get("denoise", {"type":"none"})
    t = den.get("type", "none")
    if t == "median":
        k = den.get("median", {}).get("blur_limit", 3)
        steps += [A.MedianBlur(blur_limit=k, p=1.0)]
    elif t == "gaussian":
        k = den.get("gaussian", {}).get("blur_limit", 3)
        sg = den.get("gaussian", {}).get("sigma", 0)
        steps += [A.GaussianBlur(blur_limit=k, sigma_limit=(sg, sg), p=1.0)]
    elif t == "bilateral":
        d  = den.get("bilateral", {}).get("d", 7)
        sc = den.get("bilateral", {}).get("sigma_color", 75)
        ss = den.get("bilateral", {}).get("sigma_space", 75)
        def _bilateral(im, **kwargs):
            im2 = im
            if im2.dtype == np.float64:
                im2 = im2.astype(np.float32)
            elif im2.dtype not in (np.uint8, np.float32):
                im2 = im2.astype(np.uint8)
            return cv2.bilateralFilter(im2, d=d, sigmaColor=sc, sigmaSpace=ss)
        steps += [A.Lambda(image=_bilateral, name="bilateral_filter")]
    elif t == "nlm":
        h  = den.get("nlm", {}).get("h", 7)
        tw = den.get("nlm", {}).get("template", 7)
        sw = den.get("nlm", {}).get("search", 21)
        def _nlm(im, **kwargs):
            if im.dtype == np.uint8:
                im8 = im
            else:
                im8 = (np.clip(im,0,1)*255).astype(np.uint8) if im.dtype.kind=='f' else im.astype(np.uint8)
            bgr = cv2.cvtColor(im8, cv2.COLOR_RGB2BGR)
            deno = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, tw, sw)
            return cv2.cvtColor(deno, cv2.COLOR_BGR2RGB)
        steps += [A.Lambda(image=_nlm, name="nlm")]
    # Contrast
    con = cfg.get("contrast", {"type":"none"})
    ct = con.get("type", "none")
    if ct == "clahe":
        clip = con.get("clahe", {}).get("clip_limit", 2.0)
        tile = tuple(con.get("clahe", {}).get("tile_grid_size", [8,8]))
        steps += [A.CLAHE(clip_limit=clip, tile_grid_size=tile, always_apply=True)]
    elif ct == "equalize":
        steps += [A.Equalize(always_apply=True)]
    # Gamma
    gamma = cfg.get("gamma", {"enabled": False, "gamma_value":1.0})
    if gamma.get("enabled", False):
        gv = gamma.get("gamma_value", 1.0)
        gamma_pct = int(round(gv * 100))
        steps += [A.Gamma(gamma_limit=(gamma_pct, gamma_pct), always_apply=True)]
    # Sharpen
    shp = cfg.get("sharpen", {"enabled": False})
    if shp.get("enabled", False):
        steps += [A.UnsharpMask(blur_limit=shp.get("blur_limit", 3),
                                alpha=shp.get("alpha", 0.25), p=1.0)]
    # ToFloat at end (if requested in enhancement; usually keep uint8 here).
    tf = cfg.get("to_float", {"enabled": False, "max_value":255.0})
    if tf.get("enabled", False):
        steps += [A.ToFloat(max_value=tf.get("max_value", 255.0), always_apply=True)]
    aug = A.Compose(steps)
    def _apply(im: np.ndarray):
        return aug(image=im)['image']
    return _apply

# -------------------------
# Stochastic augmentation presets
# -------------------------

def build_augmentations(preset: str, img_size=(224,224)):
    preset = (preset or "medium").lower()
    if _have_albu:
        # Albumentations branch
        import albumentations as A
        from albumentations import pytorch as AP
        if preset == "light":
            tf = A.Compose([
                A.SmallestMaxSize(max_size=img_size[0]),
                A.RandomCrop(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
                A.HueSaturationValue(10,10,5, p=0.25),
                A.CoarseDropout(max_holes=1, max_height=0.15, max_width=0.15, p=0.2),
                AP.transforms.ToTensorV2(),
            ])
        elif preset == "strong":
            tf = A.Compose([
                A.LongestMaxSize(max_size=img_size[0]),
                A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_REFLECT_101),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=25, p=0.8, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.8),
                A.OneOf([A.MotionBlur(p=0.2), A.MedianBlur(blur_limit=5, p=0.2), A.GaussianBlur(blur_limit=7, p=0.2)], p=0.4),
                A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.3), A.PiecewiseAffine(p=0.3)], p=0.3),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
                A.HueSaturationValue(20,25,10, p=0.5),
                A.RGBShift(15,15,15, p=0.2),
                A.CoarseDropout(max_holes=2, max_height=0.2, max_width=0.2, p=0.5),
                AP.transforms.ToTensorV2(),
            ])
        else:  # medium
            tf = A.Compose([
                A.SmallestMaxSize(max_size=img_size[0]),
                A.RandomCrop(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.6, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
                A.HueSaturationValue(15,20,10, p=0.4),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
                A.CoarseDropout(max_holes=1, max_height=0.15, max_width=0.15, p=0.3),
                AP.transforms.ToTensorV2(),
            ])
        return ("albu", tf)
    else:
        # torchvision fallback (saves PIL->np at the end)
        if T is None:
            raise RuntimeError("Neither Albumentations nor torchvision is available.")
        if preset == "light":
            tf = T.Compose([
                T.Resize(img_size), T.RandomCrop(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.2,0.2,0.2,0.08),
            ])
        elif preset == "strong":
            tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.RandomAdjustSharpness(1.75)], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=7)], p=0.4),
                T.ColorJitter(0.3,0.3,0.3,0.1),
                T.RandomGrayscale(p=0.07),
            ])
        else:
            tf = T.Compose([
                T.Resize(img_size), T.RandomCrop(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.RandomAdjustSharpness(1.75)], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=7)], p=0.4),
                T.ColorJitter(0.25,0.25,0.25,0.08),
                T.RandomGrayscale(p=0.05),
            ])
        return ("tv", tf)

# -------------------------
# Core runner
# -------------------------

def run(cfg: dict):
    set_all_seeds(int(cfg.get("seed", 42)))

    manifest_path = cfg["data"]["manifest_csv"]
    root_dir = cfg["data"].get("root_dir", None)  # optional prefix for relative filepaths
    splits = [s.lower() for s in cfg["data"].get("splits", ["train"])]
    output_dir = cfg["output"]["dir"]
    output_format = cfg["output"].get("format", "jpg")
    quality = int(cfg["output"].get("jpeg_quality", 95))
    n_aug = int(cfg["aug"].get("per_image", 1))
    preset = cfg["aug"].get("preset", "medium")
    img_size = tuple(cfg["aug"].get("img_size", [224,224]))
    keep_tree = bool(cfg["output"].get("keep_label_dirs", True))
    new_manifest = cfg["output"].get("manifest_csv", None)
    overwrite = bool(cfg["output"].get("overwrite", False))

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if not {'filepath','label','split'}.issubset(df.columns):
        raise ValueError("Manifest must contain 'filepath','label','split' columns.")

    # Build deterministic preprocessing
    manip_fn = build_manipulation(cfg.get("manipulation", {
        "to_float":{"enabled":False,"max_value":255.0},
        "size":{"mode":"letterbox","letterbox":{"target_hw":[640,640],"pad_value":[114,114,114]}},
    }))
    enh_fn = build_enhancement(cfg.get("enhancement", {
        "denoise":{"type":"none"},
        "contrast":{"type":"none"},
        "gamma":{"enabled":False,"gamma_value":1.0},
        "sharpen":{"enabled":False,"alpha":0.25,"blur_limit":3},
        "to_float":{"enabled":False,"max_value":255.0},
    }))

    # Build stochastic augmentations
    backend, aug_tf = build_augmentations(preset, img_size)

    records = []

    # Only work on selected splits
    work_df = df[df['split'].str.lower().isin(splits)].copy().reset_index(drop=True)

    for i, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Augmenting"):
        fp = str(row['filepath'])
        if not os.path.isabs(fp) and root_dir:
            fp = os.path.join(root_dir, fp)
        label = str(row['label'])

        # read + deterministic preprocessing
        img = load_image_cv2_rgb(fp)  # RGB uint8
        img = manip_fn(img)
        img = enh_fn(img)

        for k in range(n_aug):
            if backend == "albu":
                x = aug_tf(image=img)['image']  # tensor C,H,W in [0,1]
                if hasattr(x, "numpy"):
                    xt = (np.clip(x.permute(1,2,0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                else:
                    # shouldn't happen
                    xt = img
            else:
                # torchvision expects PIL
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        im = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8) if img.dtype.kind=='f' else img.astype(np.uint8))
                    else:
                        im = Image.fromarray(img)
                else:
                    im = img
                out_pil = aug_tf(im)
                xt = np.array(out_pil.convert('RGB'))

            # output path
            if keep_tree:
                cls_dir = os.path.join(output_dir, label)
                ensure_dir(cls_dir)
                base = Path(fp).stem
                out_fp = os.path.join(cls_dir, f"{base}_aug{k}.{output_format}")
            else:
                ensure_dir(output_dir)
                base = Path(fp).stem
                out_fp = os.path.join(output_dir, f"{label}__{base}_aug{k}.{output_format}")

            if (not overwrite) and os.path.exists(out_fp):
                # make it unique
                out_fp = os.path.join(os.path.dirname(out_fp), f"{Path(fp).stem}_aug{k}_{i}.{output_format}")

            save_image_cv2_rgb(out_fp, xt, quality=quality)
            rel_out = os.path.relpath(out_fp, root_dir) if root_dir else out_fp
            records.append({"filepath": rel_out, "label": label, "split": row['split']})

    if new_manifest:
        base_df = df.copy()
        aug_df = pd.DataFrame.from_records(records)
        out_df = pd.concat([base_df, aug_df], ignore_index=True)
        out_df.to_csv(new_manifest, index=False)
        print(f"Wrote new manifest with augmented rows: {new_manifest}")
    else:
        print("Augmentation finished. No manifest output requested.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config")
    args = parser.parse_args()

    # Parse config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}")
        sys.exit(2)

    # YAML or JSON
    if cfg_path.lower().endswith((".yml", ".yaml")):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    run(cfg)

if __name__ == "__main__":
    main()

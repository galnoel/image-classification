
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_extraction.py — minimal, config-driven feature extraction

Files:
- feature_extraction.py (this file)
- config_fe.py           # provides FE_CONFIG dict

Usage:
  python feature_extraction.py

What it does:
- Reads FE_CONFIG from config_fe.py
- Computes ONE technique across a manifest
- Writes either .npy feature vectors (features mode) or processed images (image mode)
- Produces a manifest next to the output directory
"""
from __future__ import annotations
import os, csv, sys, numpy as np, cv2
from typing import Dict, Any, Tuple

try:
    from config_fe import FE_CONFIG as CFG
except Exception as e:
    print("[FATAL] Cannot import FE_CONFIG from config_fe.py", file=sys.stderr)
    raise

# --- Techniques (same as earlier clean implementations) ---
from skimage.feature import blob_log, hog as sk_hog
from skimage.color import rgb2gray

def edge_detection(img, method="canny", **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if method == "canny":
        t1 = float(kwargs.get("threshold1", 100))
        t2 = float(kwargs.get("threshold2", 200))
        return cv2.Canny(gray, t1, t2)
    elif method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(np.hypot(gx, gy))
    elif method == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(np.hypot(gx, gy))
    elif method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(np.abs(lap))
    else:
        raise ValueError("unknown edge method")

def corner_detection(img, method="harris", max_points=500, quality=0.01, min_distance=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if method == "harris":
        gray_f32 = np.float32(gray)
        dst = cv2.cornerHarris(gray_f32, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        vis = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return vis
    elif method == "shi_tomasi":
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=quality, minDistance=min_distance)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if pts is not None:
            for (x, y) in pts.squeeze(1).astype(int):
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        return vis
    else:
        raise ValueError("unknown corner method")

def blob_detection_func(img, min_sigma=1.0, max_sigma=30.0, num_sigma=10, threshold=0.02):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    if im.ndim == 3:
        im = rgb2gray(im)
    im = im.astype(np.float32)
    im = (im - im.min()) / max(1e-6, (im.max() - im.min()))
    blobs = blob_log(im, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    vis = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (y, x, s) in blobs:
        r = int((2 ** 0.5) * s)
        cv2.circle(vis, (int(x), int(y)), r, (255, 0, 0), 1)
    return vis

def color_histogram(img, bins=32):
    chans = cv2.split(img)
    hists = [cv2.calcHist([c], [0], None, [bins], [0, 256]).flatten() for c in chans]
    feat = np.concatenate(hists).astype(np.float32)
    s = feat.sum()
    if s > 0: feat /= s
    return feat

def intensity_histogram(img, bins=32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten().astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    return hist

def hog_features(img, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    if im.ndim == 3:
        im = rgb2gray(im)
    im = im.astype(np.float32)
    feat = sk_hog(im, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False, channel_axis=None, feature_vector=True)
    return feat.astype(np.float32)

def orb_features(img, n_features=500):
    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp, des = orb.detectAndCompute(img, None)
    vis = img.copy()
    if img.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis

def grabcut(img, rect=None, iters=5):
    if rect is None:
        raise ValueError("grabcut requires rect [x,y,w,h] in config")
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

# --- Batch driver (config only) ---

VEC_TECHS = {"color_histogram","intensity_histogram","hog"}
IMG_TECHS = {"canny","sobel","scharr","laplacian","harris","shi_tomasi","log_blobs","orb","grabcut"}

def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _is_abs(p): return os.path.isabs(p)

def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    io = CFG["io"]
    manifest   = io["manifest"]
    images_root= io["images_root"]
    out_dir    = io["out_dir"]
    path_col   = io.get("path_col","filepath")
    label_col  = io.get("label_col")
    extra_cols = io.get("extra_cols", [])
    ext        = io.get("ext",".png")

    tech   = CFG["technique"]
    params = CFG.get("params", {}).get(tech, {})

    rows = _read_rows(manifest)
    if not rows:
        print("[INFO] Empty manifest; nothing to do."); return

    _ensure_dir(out_dir)

    default_mode = "features" if tech in VEC_TECHS else "image"
    out_mode = CFG.get("output_mode", default_mode)

    out_manifest = os.path.join(os.path.dirname(out_dir.rstrip("/")), f"{tech}_manifest.csv")
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

            if out_mode == "features":
                if tech == "color_histogram":
                    vec = color_histogram(img, **params)
                elif tech == "intensity_histogram":
                    vec = intensity_histogram(img, **params)
                elif tech == "hog":
                    # ensure tuple parameters
                    if "pixels_per_cell" in params: params["pixels_per_cell"] = tuple(params["pixels_per_cell"])
                    if "cells_per_block" in params: params["cells_per_block"] = tuple(params["cells_per_block"])
                    vec = hog_features(img, **params)
                else:
                    print(f"[WARN] {tech} is not a vector technique"); continue
                out_path = os.path.join(out_dir, base + ".npy")
                np.save(out_path, vec)
            else:
                if tech in {"canny","sobel","scharr","laplacian"}:
                    vis = edge_detection(img, method="canny" if tech=="canny" else tech, **params)
                elif tech in {"harris","shi_tomasi"}:
                    vis = corner_detection(img, method=tech, **params)
                elif tech == "log_blobs":
                    vis = blob_detection_func(img, **params)
                elif tech == "orb":
                    vis = orb_features(img, **params)
                elif tech == "grabcut":
                    rect = params.get("rect")
                    if rect is None: raise ValueError("config_fe.py missing params.grabcut.rect")
                    vis = grabcut(img, rect=tuple(rect), iters=params.get("iters",5))
                else:
                    print(f"[WARN] Unsupported image technique: {tech}"); continue
                out_path = os.path.join(out_dir, base + ext)
                cv2.imwrite(out_path, vis)

            row_o = {path_col: os.path.relpath(out_path, out_dir)}
            if label_col: row_o[label_col] = r.get(label_col,"")
            for c in extra_cols: row_o[c] = r.get(c,"")
            writer.writerow(row_o); wrote += 1

    print(f"[OK] Wrote {wrote} outputs using technique={tech} mode={out_mode}")
    print(f"[OK] Manifest: {out_manifest} ; out_dir: {out_dir}")


if __name__ == "__main__":
    main()

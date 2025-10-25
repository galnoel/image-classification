
# config_aug.py — single place to control augmentation I/O + toggles

AUGMENTATION_CONFIG = {
    "io": {
        "manifest":    "data/manifest.csv",
        "images_root": "data/clean_images",
        "out_dir":     "data/aug_images",
        "path_col":    "filepath",
        "label_col":   "label",     # or None if unlabeled
        "extra_cols":  [],
        "copies":      3,           # augmented copies per input
        "ext":         ".jpg",
    },
    "augmentations": {
        "seed": 42,
        "image_size": None,  # or [H, W], or None for RESIZE
        "transforms": {
            "rotate": {"enabled": True,  "limit": 15, "p": 0.5},
            "flip":   {"enabled": False, "horizontal": True, "vertical": False, "p": 0.5},
            "affine": {"enabled": False, "scale": [0.9, 1.1], "translate_percent": [0.0, 0.1], "rotate": 0, "shear": [-10, 10], "p": 0.3},
            "brightness_contrast": {"enabled": True, "brightness_limit": 0.15, "contrast_limit": 0.15, "p": 0.3},
            "color_jitter":  {"enabled": False, "brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02, "p": 0.2},
            "blur":          {"enabled": False, "blur_limit": 3, "p": 0.2},
            "gaussian_noise":{"enabled": False, "var_limit": [10.0, 50.0], "p": 0.2},
            "sharpen":       {"enabled": False, "alpha": [0.2, 0.5], "lightness": [0.5, 1.0], "p": 0.2},
            "jpeg":          {"enabled": False, "quality_lower": 80, "quality_upper": 100, "p": 0.2},
            "grayscale":     {"enabled": False, "p": 0.1},
        }
    }
}

#how to use image_size:

    # "image_size": 384,      # int → resize to 384×384 (square)
    # OR
    # "image_size": [320,480] # tuple/list → resize to H×W exactly (keeps a fixed rectangle)
    # OR
    # "image_size": None      # don’t resize; keep original size

# Extended configuration with Albumentations-only knobs
AUGMENTATION_CONFIG = {
    "data": {
        "image_size": 224,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225]
        },
        "batch_size": 32,
        "num_workers": 4,
        "pin_memory": True
    },
    "train": {
        # geometric
        "random_resized_crop": { "enabled": True,  "size": 224, "scale": [0.85, 1.0], "ratio": [0.75, 1.33] },
        "resize":              { "enabled": False, "size": None },
        "horizontal_flip":     { "enabled": True,  "p": 0.5 },
        "vertical_flip":       { "enabled": False, "p": 0.0 },
        "rotation":            { "enabled": True, "degrees": 15 },
        "affine":              { "enabled": False, "translate": 0.10, "scale": [0.9, 1.1], "shear": 15, "degrees": 0 },

        # photometric (new + old)
        "color_jitter":        { "enabled": False, "brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.02 },
        "brightness_contrast": { "enabled": False, "brightness_limit": 0.15, "contrast_limit": 0.15 },
        "blur":                { "enabled": False, "blur_limit": 3 },
        "gaussian_noise":      { "enabled": False, "var_limit": [10.0, 50.0] },
        "sharpen":             { "enabled": False, "alpha": [0.2, 0.5], "lightness": [0.5, 1.0] },
        "jpeg":                { "enabled": False, "quality_lower": 80, "quality_upper": 100 },
        "grayscale":           { "enabled": False },

        # NEW: parity with aug_detection_fix extras
        "clahe":               { "enabled": False, "clip_limit": 4.0, "tile_grid_size": [8, 8], "p": 0.5 },
        "ahe":                 { "enabled": False, "clip_limit": 8.0, "tile_grid_size": [8, 8], "p": 0.5 },
        "equalize":            { "enabled": False, "mode": "cv", "by_channels": True, "p": 0.5 },
        "denoise":             { "enabled": False, "h": 10, "hColor": 10, "templateWindowSize": 7, "searchWindowSize": 21, "p": 0.5 },
        "defog":               { "enabled": False, "omega": 0.95, "t0": 0.1, "radius": 7, "p": 0.5 }
    },
    "test": {
        "resize":       { "enabled": True,  "size": None },
        "center_crop":  { "enabled": False, "size": None }
    }
}

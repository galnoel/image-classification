# config_aug.py — single source of truth for transforms + dataloaders

AUGMENTATION_CONFIG = {
    "data": {
        # Size used by both train/test transforms (square int or [H, W])
        "image_size": 224,

        # Normalization (ImageNet defaults — works for most pretrained CNN/ViT)
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225]
        },

        # DataLoader knobs (you can override in prepare_data() call too)
        "batch_size": 32,
        "num_workers": 4,
        "pin_memory": True
    },

    # TRAIN-time augmentation (applied in order top→bottom)
    "train": {
        "random_resized_crop": { "enabled": True,  "scale": [0.85, 1.0], "ratio": [0.75, 1.33] },
        "resize":              { "enabled": False, "size": None },   # used if you turn off random_resized_crop
        "horizontal_flip":     { "enabled": True,  "p": 0.5 },
        "vertical_flip":       { "enabled": False, "p": 0.0 },
        "rotation":            { "enabled": True, "degrees": 15 },
        "affine":              { "enabled": False, "translate": 0.10, "scale": [0.9, 1.1], "shear": 15, "degrees": 0 },
        "color_jitter":        { "enabled": False, "brightness": 0.15, "contrast": 0.15,
                                                  "saturation": 0.15, "hue": 0.02 },
        "trivial_augment_wide":{ "enabled": False },                 # requires torchvision>=0.13
        "randaugment":         { "enabled": False, "num_ops": 2, "magnitude": 9 }
    },

    # TEST/VALID-time transforms (no randomness)
    "test": {
        "resize":       { "enabled": True,  "size": None },          # None => (image_size, image_size)
        "center_crop":  { "enabled": False, "size": None }
    }
}

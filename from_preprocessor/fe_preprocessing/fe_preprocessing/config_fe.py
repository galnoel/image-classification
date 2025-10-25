
# config_fe.py — single place to control feature extraction

FE_CONFIG = {
    "io": {
        "manifest":    "data/aug_manifest.csv",
        "images_root": "data/aug_images",
        "out_dir":     "data/features",
        "path_col":    "filepath",
        "label_col":   "label",
        "extra_cols":  [],
        "ext":         ".png",   # used only when outputting images
    },
    # choose ONE technique:
    "technique": "hog",  # canny, sobel, scharr, laplacian, harris, shi_tomasi, log_blobs, orb, grabcut, color_histogram, intensity_histogram, hog
    # optional (auto-inferred if omitted): "features" for vector techs, "image" for map techs
    # "output_mode": "features",
    "params": {
        "hog": {
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "orientations": 9
        },
        "canny": {
            "threshold1": 50,
            "threshold2": 150
        },
        "grabcut": {
            "rect": [10, 10, 200, 200],
            "iters": 5
        },
        "log_blobs": {
            "min_sigma": 1.0,
            "max_sigma": 30.0,
            "num_sigma": 10,
            "threshold": 0.02
        }
    }
}

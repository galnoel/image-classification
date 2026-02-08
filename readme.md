# Image Classification and Object Detection Project

This repository contains code and resources for various computer vision tasks, specifically focusing on Image Classification and Object Detection. It's structured to facilitate experimentation, training, evaluation, and prediction using different models and datasets.

## Features

-   **Image Classification:** Tools and scripts for training and evaluating image classification models on custom datasets.
-   **Object Detection:** Dedicated section for object detection tasks, likely leveraging models like YOLO or Faster R-CNN.
-   **Data Preprocessing & Augmentation:** Utilities for preparing image data, including augmentation pipelines.
-   **Experiment Tracking:** Integration with tools like Weights & Biases (wandb) for logging experiment metrics and results.
-   **Model Management:** Scripts for downloading, setting up, and managing various pretrained models.

## Folder Structure

A high-level overview of the project's directory structure:

```
.
├── config_cv.yaml               # Configuration for image classification
├── download_model.py            # Script to download pretrained models
├── environment.yml              # Conda environment definition
├── evaluate.py                  # Script for evaluating image classification models
├── explain.py                   # Script for model explanations (e.g., LIME, Grad-CAM)
├── image_classification_1.ipynb # Jupyter notebook for image classification exploration
├── oral-cancer-classification.ipynb # Specific notebook for oral cancer classification
├── predict.py                   # Script for running predictions with image classification models
├── readme.md                    # This file
├── requirements.txt             # Python dependencies
├── run_cv_pipeline.py           # Main script to run the image classification pipeline
├── set_model.py                 # Script to set up a specific model
│
├── cat_vs_dog/                  # Example dataset for Cat vs Dog classification
│   ├── train/
│   └── val/
│
├── data/                        # Contains various datasets
│   ├── histopathologic-oral-cancer/
│   └── PetImages/
│
├── from_preprocessor/           # Scripts and configs related to external preprocessing
│   ├── augment_config_modified.yaml
│   ├── augment_config.yaml
│   ├── augment_images.py
│   ├── checksums.txt
│   ├── class_map.json
│   ├── manifest_full_processed_rel.csv
│   ├── augmentation_preprocessing/
│   └── fe_preprocessing/
│
├── notebooks/                   # Collection of Jupyter notebooks for analysis and experimentation
│   ├── catvsdog.ipynb
│   ├── image_classification_f1.ipynb
│   └── image_classification.ipynb
│
├── object-detection/            # Dedicated module for Object Detection tasks
│   ├── config_od.yaml           # Configuration for object detection
│   ├── dataset.yaml
│   ├── download_yolo.py         # Script to download YOLO models
│   ├── evaluate_yolo.py         # Script to evaluate YOLO models
│   ├── predict_yolo.py          # Script to run predictions with YOLO models
│   ├── run_od_pipeline.py       # Main script to run the object detection pipeline
│   ├── train_yolo.py            # Script to train YOLO models
│   └── src/                     # Source code for object detection
│       ├── data_setup.py
│       ├── engine.py
│       ├── model_definitions.py
│       ├── torchvision_trainer.py
│       ├── train.py
│       └── utils.py
│
├── outputs/                     # Stores results, logs, checkpoints from runs
│   ├── coat_lite_mini_YYYYMMDD_HHMMSS/ # Example experiment output
│   └── efficientnet_b0_YYYYMMDD_HHMMSS/
│
├── pretrained_models/           # Directory to store downloaded pretrained models
│
├── project_mnervian/            # Specific project related notebooks
│   └── eda-preprocessing-oral-histopatologi.ipynb
│
└── src/                         # Core source code for image classification
    ├── aug_albumentation_2.py
    ├── augmentation.py          # Image augmentation utilities
    ├── config_aug_fix.py
    ├── config_aug.py            # Augmentation configuration
    ├── data_setup.py            # Data loading and preprocessing utilities
    ├── dataloader_factory.py
    ├── early_stopping.py        # Early stopping logic
    ├── engine.py                # Training and evaluation loop engine
    ├── model_definitions.py     # Model architectures and definitions
    ├── train.py                 # Main training script
    └── utils.py                 # Utility functions
```

## Setup

### Prerequisites

-   Python 3.x
-   Conda (recommended for environment management) or Pip

### 1. Clone the repository

```bash
git clone <repository-url>
cd image-classification
```

### 2. Create and activate Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate image-classification-env # Replace with your environment name
```

### 3. Install dependencies (if not using Conda or for additional packages)

```bash
pip install -r requirements.txt
```

## Usage

### Image Classification

#### Training a model

To train an image classification model, you typically need to specify a configuration file.

```bash
python run_cv_pipeline.py --config config_cv.yaml --mode train
```
*Note: Adjust the config file and parameters as per your specific training requirements.*

#### Evaluating a model

After training, you can evaluate the model's performance:

```bash
python evaluate.py --model_path outputs/your_model_run/model.pth --config config_cv.yaml
```
*Note: Replace `your_model_run/model.pth` with the actual path to your trained model checkpoint.*

#### Making Predictions

To make predictions on new images:

```bash
python predict.py --model_path outputs/your_model_run/model.pth --image_path path/to/your/image.jpg
```
*Note: Replace placeholders with actual paths.*

#### Explaining Predictions

To understand model predictions using explainability methods:

```bash
python explain.py --model_path outputs/your_model_run/model.pth --image_path path/to/your/image.jpg
```

### Object Detection

Refer to the `object-detection/` directory for specific instructions on setting up, training, and evaluating object detection models.

#### Training a YOLO model

```bash
python object-detection/train_yolo.py --config object-detection/config_od.yaml
```

#### Running YOLO predictions

```bash
python object-detection/predict_yolo.py --weights object-detection/runs/train/exp/weights/best.pt --source path/to/image_or_video
```

## Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Write appropriate tests.
5.  Ensure all tests pass.
6.  Submit a pull request.


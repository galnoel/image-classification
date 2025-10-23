# predict.py
import torch
from torch import nn
from torchvision import models, datasets
import yaml
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

from src import data_setup # Import your data setup functions

def predict(args):
    """Generates predictions and creates a submission file."""
    # 1. Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Setup logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 3. Get class names from the TRAINING data directory
    train_dir = Path(cfg['data']['train_dir'])
    temp_dataset = datasets.ImageFolder(root=train_dir)
    class_names = temp_dataset.classes
    logging.info(f"Loaded class names for mapping: {class_names}")

    # 4. Initialize Model Architecture
    model_name = cfg['model_params']['name']
    logging.info(f"Initializing model architecture: {model_name}")

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=len(class_names))
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # 5. Load the trained weights
    model_path = Path(args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Loaded trained model weights from {model_path}")

    # 6. Create the submission DataLoader
    submission_loader = data_setup.create_submission_dataloader(cfg, device)

    # 7. Generate Predictions
    logging.info("Generating predictions...")
    image_ids = []
    predictions = []
    with torch.inference_mode():
        for images, ids in tqdm(submission_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predicted_labels = [class_names[p] for p in preds.cpu().numpy()]
            image_ids.extend(ids)
            predictions.extend(predicted_labels)

    # 8. Create and Save Submission File
    submission_df = pd.DataFrame({'id': image_ids, 'label': predictions})
    submission_path = model_path.parent / "submission.csv" # Save in the same run folder
    submission_df.to_csv(submission_path, index=False)
    logging.info(f"Submission file created at: {submission_path}")
    print("\nSubmission file head:")
    print(submission_df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions for a competition.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help="Path to the trained model .pth file from a specific run."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default="config_cv.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    predict(args)
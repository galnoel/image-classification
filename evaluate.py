# evaluate.py
import torch
from torch import nn
from torchvision import models, datasets, transforms
import yaml
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report, accuracy_score

def evaluate(args):
    """Evaluates a trained model on a labeled test set."""
    # 1. Load configuration and setup
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 2. IMPORTANT: Create a DataLoader using ImageFolder
    # This loads both images and their true labels from the folder structure.
    labeled_test_dir = Path(cfg['data']['labeled_test_dir'])
    logging.info(f"Loading labeled test data from: {labeled_test_dir}")

    image_size = cfg['data_params']['image_size']
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(root=labeled_test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg['data_params']['batch_size'],
        shuffle=False,
        num_workers=cfg['data_params']['num_workers']
    )
    class_names = test_dataset.classes

    # 3. Initialize Model Architecture and Load Trained Weights
    model_path = Path(args.model_path)
    model_name = cfg['model_params']['name']
    logging.info(f"Initializing model architecture: {model_name}")

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=len(class_names))
    elif model_name == "resnet18":
        # ... (add logic for other models if needed) ...
        pass
    
    
    # --- ADD THIS LOGIC ---
    # If the user provided a directory, find the fine-tuned model inside it
    if model_path.is_dir():
        logging.info(f"Directory provided. Searching for a model file in: {model_path}")
        # Prioritize the fine-tuned model if it exists
        found_path = next(model_path.glob('*finetuned.pth'), None)
        if not found_path:
            # Otherwise, look for any .pth file
            found_path = next(model_path.glob('*.pth'), None)
        
        if found_path:
            model_path = found_path
        else:
            raise FileNotFoundError(f"No .pth model file found in the directory: {args.model_path}")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    logging.info(f"Loaded trained model weights from {model_path}")

    # 4. Generate Predictions and Collect True Labels
    logging.info("Generating predictions to evaluate...")
    all_preds = []
    all_true_labels = []
    
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # The model makes predictions without seeing the labels
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # The script "remembers" the true labels to compare
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 5. Calculate and Save Metrics
    accuracy = accuracy_score(all_true_labels, all_preds)
    report = classification_report(all_true_labels, all_preds, target_names=class_names)
    
    print("\n--- Evaluation Report ---")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(report)
    
    # Save the report to a file in the same run folder as the model
    report_path = model_path.parent / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Final Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    logging.info(f"Evaluation report saved to: {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a labeled test set.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--config', type=str, default="config_cv.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    evaluate(args)
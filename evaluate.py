# evaluate.py

#python evaluate.py --model_path outputs/efficientnet_b0.../best_model_finetuned.pth --mode eval

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

# We need the function to create the submission dataloader
from src import data_setup

def main(args):
    """
    Evaluates a model or generates predictions based on the specified mode.
    Mode 'eval': Evaluates on a labeled test set.
    Mode 'predict': Generates a submission.csv for an unlabeled test set.
    """
    # 1. Load configuration and setup
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running in '{args.mode}' mode on device: {device}")

    # 2. Initialize Model Architecture and Load Trained Weights
    model_path = Path(args.model_path)
    model_name = cfg['model_params']['name']
    
    # We need class_names to build the model head correctly. Get them from the training data folder.
    train_dir = Path(cfg['data']['train_dir'])
    class_names = datasets.ImageFolder(root=train_dir).classes
    logging.info(f"Initializing model '{model_name}' for {len(class_names)} classes: {class_names}")

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=len(class_names))
    # Add elif blocks for other models like resnet18 if you use them
    else:
        raise ValueError(f"Model architecture '{model_name}' not defined in evaluate.py")
    
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

    # 3. Process based on the selected mode
    if args.mode == 'eval':
        # --- EVALUATION MODE ---
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

        all_preds = []
        all_true_labels = []
        with torch.inference_mode():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_true_labels, all_preds)
        report = classification_report(all_true_labels, all_preds, target_names=class_names)
        
        print("\n--- Evaluation Report ---")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(report)
        
        report_path = model_path.parent / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Final Test Accuracy: {accuracy:.4f}\n\n{report}")
        logging.info(f"Evaluation report saved to: {report_path}")

    elif args.mode == 'predict':
        # --- PREDICTION (SUBMISSION) MODE ---
        submission_loader = data_setup.create_submission_dataloader(cfg, device)
        logging.info(f"Loading unlabeled data from: {cfg['data']['unlabeled_test_dir']}")

        image_ids = []
        predictions = []
        with torch.inference_mode():
            for images, ids in tqdm(submission_loader, desc="Predicting"):
                images = images.to(device)
                outputs = model(images)
                _, preds_indices = torch.max(outputs, 1)
                predicted_labels = [class_names[p] for p in preds_indices.cpu().numpy()]
                image_ids.extend(ids)
                predictions.extend(predicted_labels)

        submission_df = pd.DataFrame({'id': image_ids, 'label': predictions})
        submission_path = model_path.parent / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        logging.info(f"Submission file created at: {submission_path}")
        print("\nSubmission file head:")
        print(submission_df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model or generate predictions.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--config', type=str, default="config_cv.yaml", help="Path to the configuration file.")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['eval', 'predict'],
        help="Operation mode: 'eval' for evaluation on labeled data, 'predict' for submission on unlabeled data."
    )
    args = parser.parse_args()
    main(args)
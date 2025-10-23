# run_cv_pipeline.py
import yaml
import logging
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
import torchvision
from torchvision import models

import data_setup
import train
import utils

# import wandb

# import os
# # import weave

# os.environ['WANDB_API_KEY'] = '930bab9b9fb78035209fd1c7943709d13819f130'

# weave.init('galnoel-universitas-sam-ratulangi/image-classification') # 🐝

# @weave.op() # 🐝 Decorator to track requests

def setup_logging(log_file):
    """Sets up the logging configuration."""
    # Ensure handlers are cleared for clean logging in subsequent runs if in interactive env
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def main():
    """Main function to run the entire CV pipeline."""
    # 1. Load configuration
    with open("config_cv.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Setup run-specific output folder and logging
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_for_folder = cfg['model_params']['name']
    run_folder_name = f"{model_name_for_folder}_{run_timestamp}"

    # Base output directory (e.g., 'outputs' as per your image)
    base_output_dir = Path(cfg['outputs']['base_dir'])
    run_folder_path = base_output_dir / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    # Update all output paths in the config to point to the current run's folder
    for key, path_template in cfg['outputs'].items():
        if isinstance(path_template, str) and "{run_folder}" in path_template:
             cfg['outputs'][key] = path_template.format(run_folder=run_folder_path)

    setup_logging(cfg['outputs']['log_file'])
    logging.info("Configuration loaded and logging set up.")
    logging.info(f"All outputs for this run will be saved in: {run_folder_path}")

    # wandb.init(
    #         project="image-classification",  # Give your project a name
    #         name=run_folder_name,          # Use your timestamped folder name for easy tracking
    #         config=cfg                     # This logs your ENTIRE config file!
    #     )

    # 3. Setup device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 4. Prepare Data: Split, Augment, and Create DataLoaders
    logging.info("Starting data preparation...")
    train_loader, test_loader, class_names = data_setup.prepare_data(cfg, device)
    logging.info(f"Data preparation complete. Found classes: {class_names}")

#----------------------------------------------------------------------------------------------------------
    # 5. Initialize the Model
    # logging.info(f"Initializing model: {cfg['model_params']['name']}")
    
    # # Dynamically get model weights and adapt classifier
    # if cfg['model_params']['name'] == "efficientnet_b0":
    #     weights = models.EfficientNet_B0_Weights.DEFAULT if cfg['model_params']['pretrained'] else None
    #     model = models.efficientnet_b0(weights=weights)
    #     # Freeze base layers
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    #     # Adapt classifier
    #     num_ftrs = model.classifier[1].in_features
    #     model.classifier = nn.Sequential(
    #         nn.Dropout(p=0.2, inplace=True),
    #         nn.Linear(in_features=num_ftrs, out_features=len(class_names))
    #     )
    # elif cfg['model_params']['name'] == "resnet18":
    #     weights = models.ResNet18_Weights.DEFAULT if cfg['model_params']['pretrained'] else None
    #     model = models.resnet18(weights=weights)
    #     # Freeze base layers
    #     for param in model.parameters(): # Freezing all by default, then unfreeze head if needed
    #         param.requires_grad = False
    #     # Adapt classifier
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, len(class_names))
    # else:
    #     raise ValueError(f"Model '{cfg['model_params']['name']}' not supported or implemented.")

#----------------------------------------------------------------------------------------------------------

    # 5. Initialize the Model by loading LOCAL weights
    model_name_cfg = cfg['model_params']['name'] # Get model name from config
    logging.info(f"Initializing model: {model_name_cfg}")
    
    # Get the directory for your local .pth files from the config
    local_weights_dir = cfg['model_params']['local_weights_dir']
    local_weights_path = Path(local_weights_dir) / f"{model_name_cfg}.pth"

    # Add a check to ensure the file exists first
    if not local_weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {local_weights_path}. "
            f"Please run 'python save_model_locally.py --model_name {model_name_cfg}' first."
        )

    # Initialize the model structure based on the name from the config
    if model_name_cfg == "efficientnet_b0":
        # 1. Create an empty model "shell"
        model = models.efficientnet_b0(weights=None)
        
        # 2. Load your local weights into the shell
        model.load_state_dict(torch.load(local_weights_path))
        
        # 3. Freeze layers and adapt the head as before
        for param in model.features.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_ftrs, out_features=len(class_names))
        )

    elif model_name_cfg == "resnet18":
        # 1. Create an empty model "shell"
        model = models.resnet18(weights=None)
        
        # 2. Load your local weights into the shell
        model.load_state_dict(torch.load(local_weights_path))
        
        # 3. Freeze layers and adapt the head as before
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        
    else:
        raise ValueError(f"Model '{model_name_cfg}' is not supported.")
        
    model = model.to(device)
    logging.info(f"Model '{cfg['model_params']['name']}' initialized and moved to '{device}'.")

    # 6. Initialize Loss Function and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train_params']['learning_rate'])
    logging.info("Loss function and optimizer initialized.")

    # 7. Start Training
    logging.info("Starting model training...")
    results = train.train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        class_names=class_names # Pass class_names for confusion matrix
    )
    logging.info("Stage 1 complete. Best model from this stage is saved.")    # -----------------------------------------------------------------
    # STAGE 2: FINE-TUNING - The new code you add
    # -----------------------------------------------------------------
    logging.info("--- Starting Stage 2: Fine-Tuning ---")

    # 1. Unfreeze the top layers of the model.
    #    For EfficientNet, we can unfreeze the last few blocks in model.features
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    logging.info("Unfroze the top layers of the model.")

    # 2. Create a new optimizer with a much lower learning rate.
    #    It's important that this optimizer now sees the newly unfrozen parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    logging.info("Created new optimizer for fine-tuning with learning rate: 0.0001")
    
    # 3. Modify config for the next training run (optional but good practice)
    #    You'll want to save the final fine-tuned model to a different file.
    cfg['outputs']['model_path'] = cfg['outputs']['model_path'].replace(".pth", "_finetuned.pth")
    cfg['train_params']['epochs'] = 3 # Train for just a few more epochs

    # 4. Continue training the model (call the train function again).
    #    The model now has unfrozen layers and will be trained with the new optimizer.
    results_finetuned = train.train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        class_names=class_names
    )
    logging.info("Stage 2 (Fine-Tuning) complete.")

    logging.info("Training complete.")

    # 8. Save final report (optional, can also save plot)
    utils.save_report(results, cfg['outputs']['report_path'])
    logging.info(f"Final performance report saved to {cfg['outputs']['report_path']}")

    # 9. (Optional) Visualize results or other post-processing
    # Example: you might add code here to plot training curves if `results` contains them.
    # wandb.finish() # <-- 3. FINISH THE WANDB RUN -->
if __name__ == '__main__':
    main()
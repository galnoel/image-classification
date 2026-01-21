    # run_cv_pipeline.py
import yaml
import logging
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
import torchvision
from torchvision import models

from src import data_setup
from src import train
from src import utils
import time # <-- ADD THIS LINE
import timm

# from focal_loss.focal_loss import FocalLoss 
from kornia.losses import FocalLoss # <-- Use the Kornia import
from src import dataloader_factory
from src.config_aug_fix import AUGMENTATION_CONFIG 

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
    
    # --- 1. RECORD START TIME ---
    start_time = time.time()

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
    #----------------------------------------------------------
    # 4. Prepare Data: Split, Augment, and Create DataLoaders
    # logging.info("Starting data preparation...")
    # train_loader, val_loader, class_names = dataloader_factory.prepare_data(cfg, device)
    #------------------------------
    logging.info("Starting data preparation...")

# Get the correct paths from your main config
    train_dir_path = cfg['data']['train_dir']
    val_dir_path = cfg['data']['val_dir']

    # Import the augmentation config that the factory function needs

    # Now call the function with the correct arguments
    train_loader, val_loader, class_names = dataloader_factory.prepare_data(
        train_dir=train_dir_path,
        valid_or_test_dir=val_dir_path,
        cfg=AUGMENTATION_CONFIG,  # Pass the specific augmentation config
        num_workers=AUGMENTATION_CONFIG['data']['num_workers'] # Pass other params as needed
    )

    logging.info(f"Using training data from: {cfg['data']['train_dir']}")
    logging.info(f"Using validation data from: {cfg['data']['test_dir']}")

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
            f"Please run 'python download_model.py --model_name {model_name_cfg}' first."
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
    
    elif model_name_cfg == "resnet50":
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(local_weights_path, weights_only=True))
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))

    elif model_name_cfg == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.load_state_dict(torch.load(local_weights_path, weights_only=True))
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, len(class_names))
    
    elif model_name_cfg == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.load_state_dict(torch.load(local_weights_path, weights_only=True))
        for param in model.parameters(): param.requires_grad = False
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, len(class_names))

    elif model_name_cfg == "swin_t":
        model = models.swin_t(weights=None)
        model.load_state_dict(torch.load(local_weights_path, weights_only=True))
        for param in model.parameters(): param.requires_grad = False
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, len(class_names))
    
    elif model_name_cfg in ["maxvit_tiny", "cvt_13", "coat_lite_mini", "efficientformerv2_s0", "levit_192"]:
        # 1. Create your model with the correct number of classes, but no weights yet.
        model = timm.create_model(model_name_cfg, pretrained=False, num_classes=len(class_names))

        # 2. Load the state dictionary from your local .pth file
        state_dict = torch.load(local_weights_path, weights_only=True)

        # 3. Get the name of the final layer (e.g., 'head.weight')
        classifier_key = model.default_cfg['classifier'] # e.g., 'head' or 'fc'
        
        # 4. Remove the incompatible final layer weights from the loaded dictionary
        # This handles cases where the key is 'head.weight', 'fc.weight', etc.
        keys_to_remove = [k for k in state_dict if k.startswith(classifier_key)]
        for key in keys_to_remove:
            del state_dict[key]
        
        # 5. Load the modified dictionary into your model.
        # `strict=False` tells PyTorch it's okay that we're missing the final layer.
        model.load_state_dict(state_dict, strict=False)

        # 6. Freeze all parameters, then unfreeze the head for training.
        for param in model.parameters(): param.requires_grad = False
        # The head is automatically unfrozen because we just created it.
        # To be explicit, you can unfreeze it again.
        for param in getattr(model, classifier_key).parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Model '{model_name_cfg}' is not supported.")
        
    model = model.to(device)
    logging.info(f"Model '{cfg['model_params']['name']}' initialized and moved to '{device}'.")

    # 6. Initialize Loss Function and Optimizer
    loss_cfg = cfg['loss_function']
    loss_name = loss_cfg['name']

    if loss_name == "FocalLoss":
        # Kornia's FocalLoss takes alpha and gamma directly.
        # It expects raw logits, which our model provides.
        loss_fn = FocalLoss(**loss_cfg['params'])
        logging.info(f"Using Kornia FocalLoss with params: {loss_cfg['params']}")
    else: # Default to CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        logging.info("Using CrossEntropyLoss.")

    # --- CHOOSE OPTIMIZER FROM CONFIG ---
    optimizer_name = cfg['train_params']['optimizer']
    learning_rate = cfg['train_params']['learning_rate']
    
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        logging.info(f"Using AdamW optimizer with learning rate: {learning_rate}")
    else: # Default to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logging.info(f"Using Adam optimizer with learning rate: {learning_rate}")

    logging.info("Loss function and optimizer initialized.")

    # --- INITIALIZE SCHEDULER FROM CONFIG ---
    scheduler_cfg = cfg['train_params'].get('scheduler', {}) # Use .get() for safety
    scheduler_name = scheduler_cfg.get('name', 'None')
    scheduler = None # Initialize as None

    if scheduler_name == "CosineAnnealingLR":
        scheduler_params = scheduler_cfg.get('params', {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        logging.info(f"Using CosineAnnealingLR scheduler with params: {scheduler_params}")
    elif scheduler_name == "ReduceLROnPlateau":
        # ... (you could add logic for other schedulers here)
        pass

    logging.info("--- Starting Stage 1: Feature Extraction ---")
    
    # 7. Start Training
    results_stage1 = train.train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler, # No scheduler for stage 1 in this setup
        device=device,
        class_names=class_names
    )
    logging.info("Stage 1 (Feature Extraction) complete.")

    # Save reports and plots for Stage 1
    utils.save_report(results_stage1, cfg['outputs']['report_path'])
    utils.plot_and_save_curves(results_stage1, cfg['outputs']['training_curves_plot_path'])


    # =================================================================
    # STAGE 2: FINE-TUNING
    # =================================================================
    logging.info("--- Starting Stage 2: Fine-Tuning ---")

    # 1. Load the best model from Stage 1 to continue training
    best_model_path_stage1 = cfg['outputs']['model_path']
    model.load_state_dict(torch.load(best_model_path_stage1, weights_only=True))
    logging.info(f"Loaded best model from Stage 1: {best_model_path_stage1}")

    # 2. Unfreeze the top layers of the model.
    model_name_cfg = cfg['model_params']['name']
    
    if model_name_cfg == 'efficientnet_b0':
        # Unfreeze the last 3 blocks for EfficientNet
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        logging.info("Unfroze the top layers of EfficientNet for fine-tuning.")
        
    elif 'resnet' in model_name_cfg:
        # Unfreeze the last block (layer4) for ResNets
        for param in model.layer4.parameters():
            param.requires_grad = True
        logging.info("Unfroze layer4 of ResNet for fine-tuning.")
        
    elif 'vit' in model_name_cfg:
        # For Vision Transformer, unfreeze the last 2 encoder blocks
        for param in model.encoder.layers[-2:].parameters():
            param.requires_grad = True
        logging.info("Unfroze the top layers of Vision Transformer for fine-tuning.")
    
    elif 'convnext' in model_name_cfg:
        # For ConvNeXt, unfreeze the final stage (the last block of layers)
        for param in model.features[-1].parameters():
            param.requires_grad = True
        logging.info("Unfroze the final stage of ConvNeXt.")

    elif 'swin' in model_name_cfg:
        # For Swin Transformer, unfreeze the final stage
        for param in model.features[-1].parameters():
            param.requires_grad = True
        logging.info("Unfroze the final stage of Swin Transformer.")

    elif model_name_cfg in ["cvt_13", "efficientformerv2_s0"]:
        # Unfreeze the final stage of the model
        for param in model.stages[-1].parameters():
            param.requires_grad = True
        logging.info(f"Unfroze the final stage of {model_name_cfg}.")

    elif model_name_cfg == "coat_lite_mini":
        # 1. Unfreeze ALL parameters in the model
        for param in model.parameters():
            param.requires_grad = True
        
        # 2. Re-freeze the earliest layers (the patch embeddings)
        # These are the first layers that process the image. It's good to keep them frozen.
        for param in model.patch_embed1.parameters():
            param.requires_grad = False
            
        logging.info(f"Fine-tuning {model_name_cfg}: Unfroze all layers except patch embeddings.")

    # LeViT uses 'blocks'
    elif model_name_cfg in ["levit_192"]:        # Unfreeze the final block of the model
        for param in model.blocks[-1].parameters():
            param.requires_grad = True
        logging.info(f"Unfroze the final block of {model_name_cfg}.")

    # 3. Create a new optimizer for fine-tuning with a lower learning rate.
    finetune_lr = cfg['train_params'].get('finetune_learning_rate', 1e-4) # Get from config or use default
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
    logging.info(f"Created new AdamW optimizer for fine-tuning with learning rate: {finetune_lr}")
    
    # 4. Update config paths to point to fine-tuning outputs for the next train call
    cfg['outputs']['model_path'] = cfg['outputs']['model_path_finetuned']
    cfg['outputs']['cm_plot_path'] = cfg['outputs']['cm_plot_path_finetuned']
    cfg['train_params']['epochs'] = cfg['train_params'].get('finetune_epochs', 3) # Get from config or use default

    # 5. Continue training the model
    results_stage2 = train.train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler, # No scheduler for stage 2 in this setup
        device=device,
        class_names=class_names
    )
    logging.info("Stage 2 (Fine-Tuning) complete.")
    
    # Save reports and plots for Stage 2
    utils.save_report(results_stage2, cfg['outputs']['report_path_finetuned'])
    utils.plot_and_save_curves(results_stage2, cfg['outputs']['training_curves_plot_path_finetuned'])
    
    logging.info(f"All artifacts for the run are saved in: {run_folder_path}")

    # --- 2. RECORD END TIME AND CALCULATE DURATION ---
    end_time = time.time()
    duration_seconds = end_time - start_time
    
    # Convert seconds to a more readable format (minutes and seconds)
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    
    # --- 3. LOG THE FINAL DURATION ---
    logging.info("="*50)
    logging.info(f"PIPELINE EXECUTION COMPLETE")
    logging.info(f"Total duration: {minutes} minutes and {seconds} seconds.")
    logging.info("="*50)

    # 9. (Optional) Visualize results or other post-processing
    # Example: you might add code here to plot training curves if `results` contains them.
    # wandb.finish() # <-- 3. FINISH THE WANDB RUN -->
if __name__ == '__main__':
    main()
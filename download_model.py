# save_model_locally.py
import torch
import torchvision.models as models
import argparse
from pathlib import Path
import timm

# --- Define the available models ---
from src.model_definitions import AVAILABLE_MODELS

def save_model(model_name, save_dir="pretrained_models"):
    """Downloads a model and saves its state_dict to a local directory."""
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available. Choose from: {list(AVAILABLE_MODELS.keys())}")
        return

    # Create the save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_file = save_path / f"{model_name}.pth"
    
    if output_file.exists():
        print(f"Model '{model_name}' already exists at {output_file}. Skipping download.")
        return

    print(f"\nDownloading {model_name}...")
    try:
        # Load the model with pre-trained weights (this triggers the download to cache)
        model = AVAILABLE_MODELS[model_name](weights='DEFAULT')
        
        # Save the model's state dictionary to your specified path
        torch.save(model.state_dict(), output_file)
        
        print(f" -> Successfully saved '{model_name}' to {output_file}")
    except Exception as e:
        print(f" -> Failed to download or save {model_name}. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and save pre-trained models to a local directory.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        help="The name of the model to download (e.g., 'efficientnet_b0')."
    )
    args = parser.parse_args()
    save_model(args.model_name)
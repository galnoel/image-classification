# explain.py 
# python explain.py --model_path outputs/efficientnet_b0_20251023_193354/best_model_finetuned.pth --image_path data_processed/test/Cat/129.jpg
#python explain.py --model_path outputs/coat_lite_mini_20260118_142653/best_model_finetuned.pth --image_path data/histopathologic-oral-cancer/test/OSCC/OSCC_100x_88.jpg

import torch
from torch import nn
from torchvision import models, transforms
import yaml
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import the Captum library for XAI
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# You may need to import timm if you are explaining a timm model
import timm

def explain_prediction(args):
    """
    Generates an XAI attribution map for a single image prediction.
    """
    # 1. Load configuration and setup
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the trained model
    model_path = Path(args.model_path)
    model_name = cfg['model_params']['name']
    
    # Get class names from your training directory
    train_dir = Path(cfg['data']['train_dir'])
    class_names = [d.name for d in train_dir.iterdir() if d.is_dir()]
    class_names.sort() # Ensure consistent order
    print(f"Initializing model '{model_name}' for classes: {class_names}")

    # --- Initialize the correct model architecture (same logic as run_cv_pipeline.py) ---
    if model_name in ["efficientnet_b0", "resnet18", "resnet50", "convnext_tiny", "swin_t"]:
        model_func = getattr(models, model_name)
        model = model_func(weights=None)
    elif model_name in ["maxvit_tiny", "cvt_13", "coat_lite_mini", "efficientformerv2_s0", "levit_192"]:
        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    else:
        raise ValueError(f"Model architecture '{model_name}' not supported.")

    # Adapt the head for the specific model
    if 'efficientnet' in model_name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    elif 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    # Add logic for other models...
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded trained model weights from {model_path}")

    # 3. Load and transform the input image
    image_path = Path(args.image_path)
    image_size = cfg['data_params']['image_size']
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img = Image.open(image_path).convert("RGB")
    transformed_img = transform(img)
    input_tensor = norm_transform(transformed_img).unsqueeze(0).to(device)

    # 4. Make a prediction
    output = model(input_tensor)
    output_probs = torch.nn.functional.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output_probs, 1)
    pred_label_name = class_names[pred_label_idx.item()]
    
    print(f"\nPrediction: '{pred_label_name}' with confidence: {prediction_score.item():.4f}")

    # 5. Generate Explanation using Integrated Gradients
    integrated_gradients = IntegratedGradients(model)
    # The target is the index of the predicted class
    attributions = integrated_gradients.attribute(input_tensor, target=pred_label_idx.item())

    # 6. Visualize and Save the Explanation
    # Convert tensor back to a displayable image format
    img_to_display = transformed_img.permute(1, 2, 0).numpy()
    
    # Visualize the attributions
    fig, _ = viz.visualize_image_attr(
        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        img_to_display,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title=f"Attribution for Prediction: {pred_label_name}"
    )
    
    # Save the figure
    save_path = model_path.parent / f"explanation_{image_path.stem}.png"
    fig.savefig(save_path)
    print(f"Explanation saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate XAI explanation for a model's prediction.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image to explain.")
    parser.add_argument('--config', type=str, default="config_cv.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    explain_prediction(args)
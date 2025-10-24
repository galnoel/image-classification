# set_model.py
import yaml
from pathlib import Path

# --- This list is the single source of truth for available models ---
# Make sure these names match the keys in your run_cv_pipeline.py logic
AVAILABLE_MODELS = [
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "vit_b_16",
    "convnext_tiny",
    "swin_t",
    "maxvit_tiny",
    "cvt_13",
    "coat_lite_mini",
    "efficientformerv2_s0",
    "levit_192",
]

CONFIG_PATH = Path("config_cv.yaml")

def set_active_model():
    """
    Displays a menu of available models and updates the config file
    based on the user's choice.
    """
    print("Please select the model you want to use for the next run:")
    
    # Display the menu
    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
        print(f"  [{i}] {model_name}")

    # Get user input
    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(AVAILABLE_MODELS):
                selected_model = AVAILABLE_MODELS[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(AVAILABLE_MODELS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Update the config file
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Change the model name
        config['model_params']['name'] = selected_model
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, sort_keys=False) # sort_keys=False preserves order
            
        print(f"\n✅ Success! The config file has been updated to use: {selected_model}")
    
    except FileNotFoundError:
        print(f"Error: Could not find the config file at {CONFIG_PATH}")
    except Exception as e:
        print(f"An error occurred while updating the config file: {e}")


if __name__ == "__main__":
    set_active_model()
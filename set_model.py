    # set_model.py
import yaml
from pathlib import Path

# --- 1. IMPORT from the new central file ---
from src.model_definitions import AVAILABLE_MODELS

# --- 2. REMOVE the old, redundant list and get names from the dictionary ---
# AVAILABLE_MODELS = [ ... ] # This is no longer needed
MODEL_NAMES = list(AVAILABLE_MODELS.keys())

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
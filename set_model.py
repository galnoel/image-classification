# set_model.py
import yaml
from pathlib import Path

# 1. Import the dictionary from your central file
from src.model_definitions import AVAILABLE_MODELS

# 2. Convert the dictionary keys into a list of names
MODEL_NAMES = list(AVAILABLE_MODELS.keys())

CONFIG_PATH = Path("config_cv.yaml")

def set_active_model():
    """
    Displays a menu of available models and updates the config file.
    """
    print("Please select the model you want to use for the next run:")
    
    # Display the menu from the list of names
    for i, model_name in enumerate(MODEL_NAMES, 1):
        print(f"  [{i}] {model_name}")

    # Get user input
    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(MODEL_NAMES):
                # 3. Use the choice to get the NAME from the list
                selected_model = MODEL_NAMES[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(MODEL_NAMES)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Update the config file
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model_params']['name'] = selected_model
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
            
        print(f"\n✅ Success! The config file has been updated to use: {selected_model}")
    
    except FileNotFoundError:
        print(f"Error: Could not find the config file at {CONFIG_PATH}")
    except Exception as e:
        print(f"An error occurred while updating the config file: {e}")


if __name__ == "__main__":
    set_active_model()
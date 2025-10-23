# utils.py
from pathlib import Path
import torch
import pickle
import logging
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module, target_path: str):
    """Saves a PyTorch model state_dict to a target directory."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(obj=model.state_dict(), f=target_path)
        logging.info(f"Model saved to {target_path}")
    except Exception as e:
        logging.error(f"Error saving model to {target_path}: {e}")

def save_report(report_dict, target_path):
    """Saves a Python dictionary (e.g., training results) to a pickle file."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target_path, 'wb') as f:
            pickle.dump(report_dict, f)
        logging.info(f"Report saved to {target_path}")
    except Exception as e:
        logging.error(f"Error saving report to {target_path}: {e}")

def plot_and_save_curves(results, save_path):
    """
    Plots the training and validation accuracy/loss curves and saves the figure.
    """
    try:
        epochs = range(len(results["train_loss"]))

        plt.figure(figsize=(12, 5))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, results["train_acc"], label="Train Accuracy")
        plt.plot(epochs, results["test_acc"], label="Validation Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, results["train_loss"], label="Train Loss")
        plt.plot(epochs, results["test_loss"], label="Validation Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Close the plot to free up memory
        logging.info(f"Training curves plot saved to: {save_path}")

    except Exception as e:
        logging.error(f"Could not create training curves plot: {e}")
# utils.py
from pathlib import Path
import torch
import pickle
import logging

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
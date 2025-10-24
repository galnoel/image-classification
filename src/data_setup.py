# data_setup.py
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def prepare_data(cfg, device):
    """
    Handles data combining (if needed), splitting, and creating PyTorch DataLoaders.
    Returns: train_loader, test_loader, class_names
    """
    source_dirs = cfg['data'].get('source_dirs') # Use .get() for optional key
    train_dir_final = Path(cfg['data']['train_dir'])
    test_dir_final = Path(cfg['data']['test_dir'])
    test_split_ratio = cfg['data']['test_split_ratio']
    
    # Ensure processed data base directory exists if splitting is performed
    base_processed_dir = train_dir_final.parent
    
    # Check if final train/test directories already exist and are populated
    if train_dir_final.is_dir() and len(list(train_dir_final.glob('*/*'))) > 0 and \
       test_dir_final.is_dir() and len(list(test_dir_final.glob('*/*'))) > 0:
        logging.info("Pre-existing split data found. Skipping data splitting.")
    elif source_dirs:
        logging.info("No pre-existing split data or incomplete. Starting data preparation...")
        
        # Clean up existing processed data directories if they're empty or incomplete
        if base_processed_dir.exists():
            logging.info(f"Cleaning up existing processed data directory: {base_processed_dir}")
            shutil.rmtree(base_processed_dir)
        
        # Create base directory for processed data
        base_processed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Collect all files by class from source directories
        logging.info(f"Collecting files from source directories: {source_dirs}")
        all_files_by_class = defaultdict(list)
        
        # This handles either source_dirs containing class folders or direct image files in source_dirs
        # Assumes each item in source_dirs is either:
        # 1. A directory containing subdirectories, where subdirectories are classes (e.g., source_dir/cat/, source_dir/dog/)
        # 2. A directory that *is* a class itself (e.g., source_dir/Cat/, source_dir/Dog/)
        # For PetImages dataset, each item in source_dirs is a class folder (e.g., PetImages/Cat)
        for source_path_str in source_dirs:
            source_path = Path(source_path_str)
            if not source_path.is_dir():
                logging.warning(f"Source path '{source_path}' is not a directory or does not exist. Skipping.")
                continue

            # Assume source_path itself is a class folder (e.g., PetImages/Cat)
            class_name = source_path.name 
            logging.info(f"Found class: {class_name} in '{source_path}'")
            
            # Add all image files from this class folder
            files = list(source_path.glob('*.*')) # Get all files directly within this class folder
            all_files_by_class[class_name].extend(files)

        # 2. Perform train/test split and copy files
        logging.info(f"Splitting combined data (test ratio: {test_split_ratio}) and copying files...")
        for class_name, files in all_files_by_class.items():
            class_train_dir = train_dir_final / class_name
            class_test_dir = test_dir_final / class_name
            class_train_dir.mkdir(parents=True, exist_ok=True)
            class_test_dir.mkdir(parents=True, exist_ok=True)

            if not files:
                logging.warning(f"No files found for class '{class_name}'. Skipping split for this class.")
                continue

            train_files, test_files = train_test_split(files, test_size=test_split_ratio, random_state=42, stratify=[f.parent.name for f in files] if len(files) > 1 else None)
            
            for f in train_files:
                try:
                    shutil.copy(f, class_train_dir / f.name)
                except Exception as e:
                    logging.warning(f"Skipping potentially corrupted/unreadable file in train set: {f.name} - {e}")
            for f in test_files:
                try:
                    shutil.copy(f, class_test_dir / f.name)
                except Exception as e:
                    logging.warning(f"Skipping potentially corrupted/unreadable file in test set: {f.name} - {e}")
        
        logging.info(f"Data splitting complete. Processed data saved to: {base_processed_dir}")
    else:
        raise FileNotFoundError("No 'source_dirs' provided in config and no pre-existing split data found. Cannot proceed.")

    # 3. Create PyTorch DataLoaders
    logging.info("Creating PyTorch DataLoaders...")
    image_size = cfg['data_params']['image_size']
    batch_size = cfg['data_params']['batch_size']
    num_workers = cfg['data_params']['num_workers']

    # Standard normalization values for models pre-trained on ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # Advanced augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_data = datasets.ImageFolder(train_dir_final, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir_final, transform=test_transform)

    class_names = train_data.classes
    logging.info(f"Classes identified: {class_names}")

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"), # Pin memory only if using GPU
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    return train_dataloader, test_dataloader, class_names

# --- NEW: Custom Dataset for Unlabeled Data ---
class UnlabeledImageDataset(Dataset):
    """Dataset for inference on unlabeled images."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.file_paths = [p for p in self.data_dir.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = Image.open(file_path).convert("RGB")
        image_id = file_path.name # Use the filename as the ID

        if self.transform:
            image = self.transform(image)
            
        return image, image_id

# --- NEW: Function to create the DataLoader for submission ---
def create_submission_dataloader(cfg, device):
    """Creates a DataLoader for the unlabeled test set for submission."""
    unlabeled_dir = cfg['data']['unlabeled_test_dir']
    if not Path(unlabeled_dir).exists():
        raise FileNotFoundError(f"Unlabeled test directory not found at: {unlabeled_dir}")

    image_size = cfg['data_params']['image_size']
    batch_size = cfg['data_params']['batch_size']
    num_workers = cfg['data_params']['num_workers']

    submission_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    submission_dataset = UnlabeledImageDataset(data_dir=unlabeled_dir, transform=submission_transform)
    
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=batch_size,
        shuffle=False, # Never shuffle test data
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    return submission_loader
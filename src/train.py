# train.py
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from . import engine
from . import utils
from .early_stopping import EarlyStopping

def train(cfg, model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, class_names):
    """The main training loop function."""
    results = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "test_loss": [], "test_acc": [], "test_f1": []
    }
    epochs = cfg['train_params']['epochs']
    model_path = cfg['outputs']['model_path']
    cm_plot_path = cfg['outputs']['cm_plot_path']

    # Initialize Early Stopping
    es_params = cfg['train_params'].get('early_stopping', {'patience': 3, 'delta': 0})
    early_stopping = EarlyStopping(
        patience=es_params.get('patience', 3),
        verbose=True,
        delta=es_params.get('delta', 0),
        path=model_path
    )
    best_test_acc = -1.0 # Initialize with a low value
    best_val_loss = float('inf')

    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_acc, train_f1 = engine.train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc, test_f1 = engine.test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device
        )

        if scheduler:
            scheduler.step()

        log_message = (
          f"Epoch: {epoch+1:02d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
        )
        if scheduler:
            # Optionally log the current learning rate to see the scheduler working
            log_message += f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
        logging.info(log_message)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)

        # Save the best model based on validation accuracy
        # if test_acc > best_test_acc:
        #     best_test_acc = test_acc
        #     utils.save_model(model, model_path)
        #     logging.info(f"New best model saved to {model_path} (Test Acc: {best_test_acc:.4f})")
        # if test_loss < best_val_loss:
        #     best_val_loss = test_loss
        #     utils.save_model(model, model_path)
        #     logging.info(f"New best model saved to {model_path} (Test Loss: {best_val_loss:.4f})")

        # Early Stopping check
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Stopping training.")
            break
        
    logging.info("Training process finished.")

    # After training, evaluate on the test set one last time with the best model (optional)
    # Or simply load the best model and run prediction to get final metrics and CM
    logging.info("Generating final confusion matrix on the test set with the best model...")
    best_model_state_dict = torch.load(model_path)
    model.load_state_dict(best_model_state_dict)
    model.eval()
    
    y_preds = []
    y_true = []
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            y_preds.extend(test_pred_labels.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix for Best Model")
    plt.savefig(cm_plot_path)
    logging.info(f"Confusion matrix saved to {cm_plot_path}")
    
    return results
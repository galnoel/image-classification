# engine.py
import torch
from tqdm import tqdm # Use standard tqdm for console output
from sklearn.metrics import f1_score, accuracy_score

def train_step(model, dataloader, loss_fn, optimizer, device):
    """Performs a single training step (one epoch)."""
    model.train()
    train_loss, train_acc = 0, 0

    all_preds = []
    all_targets = []

    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Move to CPU for sklearn calculation
        all_preds.extend(y_pred_class.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    train_f1 = f1_score(all_targets, all_preds, average='weighted')
    return train_loss, train_acc, train_f1

def test_step(model, dataloader, loss_fn, device):
    """Performs a single testing step (one epoch)."""
    model.eval()
    test_loss, test_acc = 0, 0

    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader, desc="Testing ")):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            all_preds.extend(test_pred_labels.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    return test_loss, test_acc, test_f1
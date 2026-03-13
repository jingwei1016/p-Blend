# train.py
import torch

# Auto-select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=20, model_save_path="best_model.pth"):
    """
    Standard single-input training loop with best-checkpoint saving.
    Inputs are permuted to (B, C, T) to fit Conv1d/Transformer.
    """
    model = model.to(DEVICE)
    if hasattr(criterion, "to"):
        try:
            criterion = criterion.to(DEVICE)
        except Exception:
            pass

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            inputs = inputs.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Loss: {train_loss:.4f}", end="")

        # 1. Evaluate on Validation set (used to select the best model)
        val_acc = evaluate_model(model, val_loader, prefix="VAL")
        
        # 2. Evaluate on Test set (for observation only, does not affect best model selection)
        if test_loader is not None:
            evaluate_model(model, test_loader, prefix="TEST")

        # 3. Save the best model based on Validation Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> [BEST] Val Acc: {best_val_acc:.2f}% -> Saved!")
        else:
            print() # Newline

    print(f"[DONE] Best Validation Accuracy: {best_val_acc:.2f}%")


def evaluate_model(model, data_loader, prefix="EVAL"):
    """Compute accuracy (%) on a dataloader; permute inputs to (B, C, T)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.permute(0, 2, 1)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / max(1, total)
    print(f" | [{prefix}] Acc: {acc:.2f}%", end="")
    model.train() # Restore train mode
    return acc
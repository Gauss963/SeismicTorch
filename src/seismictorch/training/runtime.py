"""Device selection and shared training/evaluation epoch execution."""

import torch


def select_device(requested):
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        requested = next(name for name, ready in available.items() if ready)
    if not available[requested]:
        raise ValueError(f"Requested device {requested} is unavailable")
    return torch.device(requested)

def run_epoch(model, loader, device, criterion, optimizer=None, train_mode=True):
    training = optimizer is not None
    model.train(training and train_mode)
    losses, predictions, targets = [], [], []

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.detach() * len(labels))
            predictions.append(logits.detach().argmax(dim=1))
            targets.append(labels)

    size = len(loader.dataset)
    mean_loss = torch.stack(losses).sum().item() / size
    if not torch.isfinite(torch.tensor(mean_loss)):
        raise FloatingPointError("Non-finite cross-entropy loss")
    # Count on CPU once per epoch. Rows are true classes, columns predictions.
    pairs = (3 * torch.cat(targets) + torch.cat(predictions)).cpu()
    matrix = torch.bincount(pairs, minlength=9).reshape(3, 3)
    accuracy = matrix.diag().sum().item() / size
    recall = matrix.diag() / matrix.sum(dim=1).clamp_min(1)
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "balanced_accuracy": recall.mean().item(),
        "confusion_matrix": matrix.tolist(),
    }

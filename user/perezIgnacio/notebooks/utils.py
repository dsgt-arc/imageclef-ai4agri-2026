import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def valid_mask(targets):
    return targets >= 0

def ordinal_predict(logits):
    return (logits > 0).sum(dim=1)

def label_to_ordinal(targets, num_classes=5):
    """
    targets: [B, H, W] with values 0..4 (or -1)
    returns: [B, K-1, H, W]
    """
    device = targets.device

    thresholds = torch.arange(1, num_classes, device=device).view(1, -1, 1, 1)

    targets_expanded = targets.unsqueeze(1)  # [B,1,H,W]

    ordinal = (targets_expanded >= thresholds).float()

    return ordinal

def ordinal_target(y, K=5):
    device = y.device
    thresholds = torch.arange(K-1, device=device).view(1, K-1, 1, 1)
    y = y.unsqueeze(1)

    return (y > thresholds).float()

def accuracy_exact(logits, targets):
    preds = ordinal_predict(logits)
    mask = valid_mask(targets)

    correct = ((preds == targets) & mask)

    return correct.sum() / mask.sum()

def accuracy_pm1(logits, targets):
    preds = ordinal_predict(logits)
    mask = valid_mask(targets)

    correct = ((torch.abs(preds - targets) <= 1) & mask)

    return correct.sum() / mask.sum()

def ordinal_loss(logits, targets, num_classes=5):
    mask = (targets >= 0).float().unsqueeze(1)

    targets_ord = label_to_ordinal(targets, num_classes)

    loss = F.binary_cross_entropy_with_logits(
        logits, targets_ord, reduction='none'
    )

    loss = loss * mask

    return loss.sum() / mask.sum()

def ordinal_confidence(logits):
    conf = torch.abs(logits).min(dim=1).values
    return torch.tanh(conf)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, label, _ in loader:
            data, label = data.to(device), label.to(device)
            
            if valid_mask(label).sum().item() == 0:
                continue

            logits = model(data)
            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses,   label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses,   label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (log scale)')
    plt.title('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {save_path}")

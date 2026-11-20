"""Training loop and evaluation metrics (top-K accuracy, spectral efficiency)."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def make_loader(X, y, batch_size=256, shuffle=True):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=True)


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, lr=1e-3, batch_size=256,
                device='cuda', noise_std=0.0, verbose=True):
    """
    Train one model. If noise_std > 0, injects Gaussian noise into training
    inputs each epoch (data augmentation — the key for Angle 5 improvement).
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    best_top1 = 0.0
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_std > 0:
                # add Gaussian noise relative to input std for scale-invariance
                scale = xb.std() * noise_std
                xb = xb + torch.randn_like(xb) * scale
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)
        sched.step()

        val_metrics = evaluate_model(model, X_val, y_val, device=device)
        history.append({
            'epoch': ep,
            'train_loss': tr_loss / tr_total,
            'train_top1': tr_correct / tr_total,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })
        if val_metrics['top1'] > best_top1:
            best_top1 = val_metrics['top1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f'  ep {ep:3d}  train_loss={tr_loss/tr_total:.4f}  '
                  f'val_top1={val_metrics["top1"]:.3f}  '
                  f'val_top3={val_metrics["top3"]:.3f}  '
                  f'val_top5={val_metrics["top5"]:.3f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_model(model, X, y, device='cuda', noise_std=0.0, batch_size=512):
    """Returns top-K accuracy for K in {1,3,5,10} and effective spectral efficiency."""
    model.eval()
    model = model.to(device)
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)

    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if noise_std > 0:
            scale = xb.std() * noise_std
            xb = xb + torch.randn_like(xb) * scale
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_y.append(yb)
    logits = torch.cat(all_logits)
    y_t = torch.cat(all_y)

    metrics = {}
    for k in (1, 3, 5, 10):
        topk = logits.topk(k, dim=1).indices
        correct = (topk == y_t.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f'top{k}'] = correct
    # Effective spectral efficiency: ratio of training only top-K beams vs all beams.
    # Here we approximate as top-K accuracy — a predicted beam in top-K means the
    # user still achieves optimal SE after refining within the K candidates.
    metrics['SE_topK_5'] = metrics['top5']
    return metrics

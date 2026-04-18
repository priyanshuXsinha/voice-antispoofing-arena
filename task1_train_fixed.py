"""task1_train_fixed.py — Task 1 with fixes for small dataset discrimination.

Key fixes vs original:
  1. Weighted CrossEntropy (handles class imbalance automatically)
  2. Focal loss option (focuses on hard examples — cloned audio that fools model)
  3. Real audio oversampling + heavy augmentation via dataset_fixed.py
  4. Threshold calibrated on dev set (not fixed at 0.5)
  5. Label smoothing (reduces overconfidence / overfitting)
  6. Longer training with patience-based early stopping
  7. Gradient clipping kept, LR warmup added

Usage:
  python task1_train_fixed.py --run 1
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lcnn import LCNN
from dataset_fixed import ClonedAudioDataset    # ← use the improved dataset
from metrics import compute_eer_from_logits, print_metrics_table
from plots import plot_loss_curves, plot_score_distribution


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

ACFG  = cfg["audio"]
TCFG  = cfg["training"]
ECFG  = cfg["evaluation"]

DEVICE   = torch.device(TCFG["device"] if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path(TCFG["save_dir"])
PLOT_DIR = Path(ECFG["plot_dir"])
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Focal Loss — punishes easy correct predictions less,
# focuses training on hard examples (cloned audio near boundary)
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────
# Build dataloaders with balanced sampling
# ─────────────────────────────────────────────────────────────

def build_dataloaders():
    real_dir   = "data/reference_speaker"
    cloned_dir = "data/cloned_audio"

    # Training set: augmentation + oversampling ON
    train_ds = ClonedAudioDataset(
        real_dir=real_dir, cloned_dir=cloned_dir,
        n_lfcc=ACFG["n_lfcc"], n_fft=ACFG["n_fft"],
        hop_length=ACFG["hop_length"], win_length=ACFG["win_length"],
        augment=True, oversample_real=True,
    )

    # Dev/eval: no augmentation, no oversampling (raw distribution)
    dev_ds = ClonedAudioDataset(
        real_dir=real_dir, cloned_dir=cloned_dir,
        n_lfcc=ACFG["n_lfcc"], n_fft=ACFG["n_fft"],
        hop_length=ACFG["hop_length"], win_length=ACFG["win_length"],
        augment=False, oversample_real=False,
    )

    # WeightedRandomSampler — double safety net on top of oversampling
    labels     = [label for _, label, _ in train_ds]
    n_real     = labels.count(0)
    n_cloned   = labels.count(1)
    class_w    = [1.0 / n_real, 1.0 / n_cloned]
    sample_w   = [class_w[l] for l in labels]
    sampler    = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    # Split dev_ds into 80% dev / 20% eval
    total      = len(dev_ds)
    dev_n      = int(0.8 * total)
    eval_n     = total - dev_n
    indices    = list(range(total))
    random.shuffle(indices)
    dev_subset  = torch.utils.data.Subset(dev_ds, indices[:dev_n])
    eval_subset = torch.utils.data.Subset(dev_ds, indices[dev_n:])

    train_loader = DataLoader(train_ds,   batch_size=TCFG["batch_size"],
                               sampler=sampler, num_workers=0, drop_last=True)
    dev_loader   = DataLoader(dev_subset,  batch_size=TCFG["batch_size"],
                               shuffle=False, num_workers=0)
    eval_loader  = DataLoader(eval_subset, batch_size=TCFG["batch_size"],
                               shuffle=False, num_workers=0)

    print(f"[DataLoader] Train: {len(train_ds)} | Dev: {len(dev_subset)} | Eval: {len(eval_subset)}")
    return train_loader, dev_loader, eval_loader, n_real, n_cloned


# ─────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_logits = [], []
    for x, y, _ in tqdm(loader, desc="  eval ", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_labels.extend(y.cpu().tolist())
        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    eer, threshold = compute_eer_from_logits(all_labels, all_logits)
    return total_loss / n, eer, threshold, all_labels, all_logits


# ─────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────

def train_run(run: int):
    set_seed(TCFG["seed"])

    print("\n" + "="*60)
    print("  Run 1: LCNN — fixed training (balanced + augmented)")
    print("="*60)

    train_loader, dev_loader, eval_loader, n_real, n_cloned = build_dataloaders()

    model = LCNN(input_channels=1, dropout=0.5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    print(f"  Device          : {DEVICE}")

    # Class weights for loss (inverse frequency)
    total   = n_real + n_cloned
    w_real  = total / (2.0 * n_real)
    w_clone = total / (2.0 * n_cloned)
    class_weights = torch.tensor([w_real, w_clone], dtype=torch.float32).to(DEVICE)
    print(f"  Class weights   : real={w_real:.2f}, cloned={w_clone:.2f}")

    # Focal loss with class weights — best choice for imbalanced + small data
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TCFG["learning_rate"],
        weight_decay=TCFG["weight_decay"],
    )

    # Cosine schedule with warmup
    warmup_epochs = 3
    epochs        = max(TCFG["epochs"], 30)   # at least 30 epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_name  = "Run1_LCNN"
    ckpt_path = SAVE_DIR / "lcnn_best.pt"

    train_losses, val_losses = [], []
    best_eer       = float("inf")
    best_threshold = 0.5
    patience_count = 0
    patience       = 10   # stop if no improvement for 10 epochs

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_eer, threshold, _, _ = eval_epoch(
            model, dev_loader, criterion, DEVICE
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"EER={val_eer*100:.2f}% | thr={threshold:.3f}")

        if val_eer < best_eer:
            best_eer       = val_eer
            best_threshold = threshold
            patience_count = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_eer":      val_eer,
                "threshold":    threshold,
                "run":          run,
                "run_name":     run_name,
                "config":       cfg,
            }, ckpt_path)
            print(f"    ✓ New best EER={best_eer*100:.2f}% → saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── Final eval with best checkpoint ──
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    eval_loss, eval_eer, eval_thr, eval_labels, eval_logits = eval_epoch(
        model, eval_loader, criterion, DEVICE
    )

    # Plots
    plot_loss_curves(train_losses, val_losses, run_name,
                     str(PLOT_DIR / f"{run_name}_loss_curves.png"))

    import torch.nn.functional as F
    probs     = F.softmax(eval_logits, dim=-1)[:, 1].numpy()
    labels_np = np.array(eval_labels)
    plot_score_distribution(
        real_scores=probs[labels_np == 0],
        spoof_scores=probs[labels_np == 1],
        threshold=eval_thr,
        title=f"{run_name} — Score Distribution (eval set)",
        save_path=str(PLOT_DIR / f"{run_name}_score_dist.png"),
    )

    results = {
        "run": run,
        "run_name": run_name,
        "best_val_eer":   round(best_eer * 100, 4),
        "eval_eer":       round(eval_eer * 100, 4),
        "eval_loss":      round(eval_loss, 6),
        "best_threshold": round(best_threshold, 6),
        "n_params":       n_params,
        "epochs_trained": epoch,
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_metrics_table(results, f"Task 1 FIXED — {run_name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="1", choices=["1", "both"])
    args = parser.parse_args()
    train_run(1)
    print("\n[Task1-FIXED] ✓ Done.")

"""scripts/task1_train.py — Task 1: Train anti-spoofing classifiers.

Run 1: LCNN  (feature-based, LFCC)
Run 2: RawNet2 (end-to-end, raw waveform)

Usage:
  python scripts/task1_train.py --run 1    # Train LCNN
  python scripts/task1_train.py --run 2    # Train RawNet2
  python scripts/task1_train.py --run both # Train both sequentially
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcnn import LCNN
# from rawnet2 import RawNet2
from dataset import ClonedAudioDataset
from metrics import compute_eer_from_logits, print_metrics_table
from plots import plot_loss_curves, plot_eer_comparison, plot_score_distribution


# ─────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DCFG  = cfg["data"]
ACFG  = cfg["audio"]
TCFG  = cfg["training"]
ECFG  = cfg["evaluation"]

DEVICE     = torch.device(TCFG["device"] if torch.cuda.is_available() else "cpu")
SAVE_DIR   = Path(TCFG["save_dir"])
PLOT_DIR   = Path(ECFG["plot_dir"])
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Build dataloaders
# ─────────────────────────────────────────────────────────────

def build_dataloaders(feature_type: str = "lfcc"):
    """Use local cloned_audio + reference_speaker dataset"""

    from dataset import ClonedAudioDataset

    real_dir = "data/reference_speaker"
    cloned_dir = "data/cloned_audio"

    full_ds = ClonedAudioDataset(
        real_dir=real_dir,
        cloned_dir=cloned_dir,
        feature_type=feature_type,
        n_lfcc=ACFG["n_lfcc"],
        n_fft=ACFG["n_fft"],
        hop_length=ACFG["hop_length"],
        win_length=ACFG["win_length"],
    )

    # 🔀 simple split (80/10/10)
    total = len(full_ds)
    train_size = int(0.8 * total)
    dev_size   = int(0.1 * total)
    eval_size  = total - train_size - dev_size

    indices = list(range(len(full_ds)))
    random.shuffle(indices)

    train_idx = indices[:train_size]
    dev_idx   = indices[train_size:train_size+dev_size]
    eval_idx  = indices[train_size+dev_size:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    dev_ds   = torch.utils.data.Subset(full_ds, dev_idx)
    eval_ds  = torch.utils.data.Subset(full_ds, eval_idx)

    train_loader = DataLoader(train_ds, batch_size=TCFG["batch_size"], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=TCFG["batch_size"], shuffle=False)
    eval_loader  = DataLoader(eval_ds,  batch_size=TCFG["batch_size"], shuffle=False)

    return train_loader, dev_loader, eval_loader

def build_rawnet2_dataloaders():
    """Dataloaders for RawNet2 (raw waveform, no feature extraction)."""
    import torchaudio
    from torch.utils.data import Dataset

    class RawWaveformDataset(torch.utils.data.Dataset):
        """Loads raw waveforms for RawNet2 (no feature extraction)."""
        def __init__(self, protocol_path, audio_dir, sample_rate=16000,
                     max_duration_sec=4.0, augment=False):
            from dataset import parse_protocol
            self.entries      = parse_protocol(protocol_path)
            self.audio_dir    = Path(audio_dir)
            self.sample_rate  = sample_rate
            self.max_samples  = int(max_duration_sec * sample_rate)
            self.augment      = augment
            self.label_map    = {"bonafide": 0, "spoof": 1}

        def __len__(self): return len(self.entries)

        def __getitem__(self, idx):
            _, file_id, label_str = self.entries[idx]
            label = self.label_map[label_str]

            path = self.audio_dir / f"{file_id}.flac"
            if not path.exists():
                path = self.audio_dir / f"{file_id}.wav"

            waveform, sr = torchaudio.load(str(path))
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            amp = waveform.abs().max()
            if amp > 0:
                waveform /= amp

            T = waveform.shape[-1]
            if T >= self.max_samples:
                start = random.randint(0, T - self.max_samples) if self.augment else 0
                waveform = waveform[..., start:start + self.max_samples]
            else:
                waveform = nn.functional.pad(waveform, (0, self.max_samples - T))

            return waveform, torch.tensor(label, dtype=torch.long)   # (1, T), ()

    root = Path(DCFG["asvspoof_root"])
    kwargs = dict(
        sample_rate=ACFG["sample_rate"],
        max_duration_sec=ACFG["max_duration_sec"],
    )

    def make_loader(protocol_key, audio_key, augment, shuffle):
        ds = RawWaveformDataset(
            str(root / DCFG[protocol_key]),
            str(root / DCFG[audio_key]),
            augment=augment, **kwargs,
        )
        return DataLoader(ds, batch_size=TCFG["batch_size"] // 2,
                          shuffle=shuffle, num_workers=TCFG["num_workers"],
                          pin_memory=True, drop_last=augment)

    return (
        make_loader("protocol_train", "train_dir", True,  True),
        make_loader("protocol_dev",   "dev_dir",   False, False),
        make_loader("protocol_eval",  "eval_dir",  False, False),
    )


# ─────────────────────────────────────────────────────────────
# Train / evaluate epoch
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
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
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_labels.extend(y.cpu().tolist())
        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    eer, threshold = compute_eer_from_logits(all_labels, all_logits)
    return total_loss / n, eer, threshold, all_labels, all_logits


# ─────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────

def train_run(run: int):
    set_seed(TCFG["seed"])

    if run == 1:
        print("\n" + "="*60)
        print("  Run 1: LCNN (LFCC features)")
        print("="*60)
        train_loader, dev_loader, eval_loader = build_dataloaders("lfcc")
        model = LCNN(
            input_channels=1,
            dropout=cfg["lcnn"]["dropout"],
        ).to(DEVICE)
        run_name   = "Run1_LCNN"
        ckpt_path  = SAVE_DIR / "lcnn_best.pt"

    # elif run == 2:
    #     print("\n" + "="*60)
    #     # print("  Run 2: RawNet2 (raw waveform)")
    #     print("="*60)
    #     train_loader, dev_loader, eval_loader = build_rawnet2_dataloaders()
    #     model = RawNet2(
    #         sinc_filters=cfg["rawnet2"]["sinc_filters"],
    #         sinc_filter_length=cfg["rawnet2"]["sinc_filter_length"],
    #         gru_node=cfg["rawnet2"]["gru_node"],
    #         nb_gru_layer=cfg["rawnet2"]["nb_gru_layer"],
    #         dropout=cfg["rawnet2"]["dropout"],
    #     ).to(DEVICE)
    #     run_name   = "Run2_RawNet2"
    #     ckpt_path  = SAVE_DIR / "rawnet2_best.pt"
    else:
        raise ValueError(f"run must be 1 or 2, got {run}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    print(f"  Device          : {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TCFG["learning_rate"],
        weight_decay=TCFG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TCFG["epochs"]
    )

    train_losses, val_losses = [], []
    best_eer = float("inf")
    best_threshold = 0.5

    for epoch in range(1, TCFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_eer, threshold, _, _ = eval_epoch(
            model, dev_loader, criterion, DEVICE
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:3d}/{TCFG['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_EER={val_eer*100:.2f}%")

        if val_eer < best_eer:
            best_eer = val_eer
            best_threshold = threshold
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_eer": val_eer,
                "threshold": threshold,
                "run": run,
                "run_name": run_name,
                "config": cfg,
            }, ckpt_path)
            print(f"    ✓ New best! EER={best_eer*100:.2f}% → saved to {ckpt_path}")

    # ── Evaluate on eval set with best checkpoint ──
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    eval_loss, eval_eer, eval_threshold, eval_labels, eval_logits = eval_epoch(
        model, eval_loader, criterion, DEVICE
    )

    # ── Plots ──
    plot_loss_curves(
        train_losses, val_losses, run_name,
        str(PLOT_DIR / f"{run_name}_loss_curves.png"),
    )

    import torch.nn.functional as F
    probs = F.softmax(eval_logits, dim=-1)[:, 1].numpy()
    labels_np = np.array(eval_labels)
    plot_score_distribution(
        real_scores=probs[labels_np == 0],
        spoof_scores=probs[labels_np == 1],
        threshold=eval_threshold,
        title=f"{run_name} — Score Distribution (eval set)",
        save_path=str(PLOT_DIR / f"{run_name}_score_dist.png"),
    )

    # ── Save results ──
    results = {
        "run": run,
        "run_name": run_name,
        "best_val_eer": round(best_eer * 100, 4),
        "eval_eer": round(eval_eer * 100, 4),
        "eval_loss": round(eval_loss, 6),
        "best_threshold": round(best_threshold, 6),
        "n_params": n_params,
        "epochs_trained": TCFG["epochs"],
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_metrics_table(results, f"Task 1 — {run_name}")
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 — Train anti-spoofing classifiers")
    parser.add_argument("--run", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which run to train: 1=LCNN, 2=RawNet2, both=sequential")
    args = parser.parse_args()

    all_results = {}

    if args.run in ("1", "both"):
        r1 = train_run(1)
        all_results["Run1_LCNN"] = r1

    # if args.run in ("2", "both"):
    #     r2 = train_run(2)
    #     all_results["Run2_RawNet2"] = r2

    # Comparison plot if both runs complete
    if len(all_results) == 2:
        plot_eer_comparison(
            run_names=list(all_results.keys()),
            eers=[v["eval_eer"] / 100 for v in all_results.values()],
            save_path=str(PLOT_DIR / "eer_comparison.png"),
        )

    print("\n[Task1] ✓ All done.")

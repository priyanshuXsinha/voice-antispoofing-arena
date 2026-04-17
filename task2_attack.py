"""task2_attack_fixed.py — Task 2: Voice cloning attack + similarity analysis (FIXED).

Fixes:
  1. Added missing `from speaker_encoder import SpeakerEncoder`
  2. Added F1 / accuracy metrics alongside EER (EER is unreliable with n=11 real)
  3. Speaker similarity analysis is now active (not commented out)

Usage:
  python task2_attack_fixed.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ✅ FIX 1: Import that was missing
from speaker_encoder import SpeakerEncoder

from lcnn import LCNN
from dataset import ClonedAudioDataset
from metrics import (
    compute_eer, compute_eer_from_logits,
    mean_cosine_similarity, print_metrics_table,
)
from plots import (
    plot_cosine_similarity_distribution,
    plot_attack_eer_bar,
    plot_score_distribution,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DCFG     = cfg["data"]
ACFG     = cfg["audio"]
ECFG     = cfg["evaluation"]
DEVICE   = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
PLOT_DIR = Path(ECFG["plot_dir"])
SAVE_DIR = Path(cfg["training"]["save_dir"])
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────

def load_model(run: int):
    ckpt_path = SAVE_DIR / "lcnn_best.pt"
    model     = LCNN(input_channels=1, dropout=0.0)
    run_name  = "Run1_LCNN"

    if not ckpt_path.exists():
        print(f"[Task2] WARNING: checkpoint not found at {ckpt_path}")
        return model.to(DEVICE), 0.5, run_name

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    threshold = ckpt.get("threshold", 0.5)
    print(f"[Task2] Loaded {run_name} from {ckpt_path} (EER threshold={threshold:.4f})")
    return model.to(DEVICE).eval(), threshold, run_name


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataset, run: int):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    all_labels, all_scores, all_paths = [], [], []

    for feats, labels, paths in tqdm(loader, desc="  inference"):
        feats = feats.to(DEVICE)
        logits = model(feats)
        probs  = F.softmax(logits, dim=-1)[:, 1]   # P(spoof)
        all_labels.extend(labels.tolist())
        all_scores.extend(probs.cpu().tolist())
        all_paths.extend(paths)

    return np.array(all_labels), np.array(all_scores), all_paths


# ─────────────────────────────────────────────────────────────
# ✅ FIX 2: Additional metrics for small/imbalanced datasets
# ─────────────────────────────────────────────────────────────

def compute_rich_metrics(labels, scores, threshold, run_name):
    """
    EER is unreliable when n_real is very small (e.g. 11 vs 50).
    Report F1, accuracy, and per-class breakdown alongside EER.
    """
    predictions = (scores >= threshold).astype(int)

    # EER (may be 0.0 if classes are trivially separable)
    eer, eer_thr = compute_eer(labels, scores)

    # Classification metrics
    acc  = accuracy_score(labels, predictions)
    f1   = f1_score(labels, predictions, average="binary", pos_label=1)
    f1_w = f1_score(labels, predictions, average="weighted")

    # Bypass: % of cloned (spoof=1) samples predicted as real (0)
    spoof_mask = labels == 1
    bypass = float(((scores[spoof_mask]) < threshold).mean()) if spoof_mask.sum() > 0 else 0.0

    print(f"\n[Task2] {run_name} — Detailed metrics (n_real={int((labels==0).sum())}, n_clone={int((labels==1).sum())})")
    print(classification_report(labels, predictions, target_names=["real", "cloned"]))

    return {
        "run_name":    run_name,
        "attack_eer":  round(eer * 100, 4),
        "accuracy":    round(acc * 100, 2),
        "f1_cloned":   round(f1  * 100, 2),
        "f1_weighted": round(f1_w * 100, 2),
        "n_real":      int((labels == 0).sum()),
        "n_cloned":    int((labels == 1).sum()),
        "pct_cloned_predicted_real": round(bypass * 100, 2),
        "note": "EER=0 is expected with small datasets; use accuracy/F1 for robustness."
                if eer == 0.0 else "",
    }


# ─────────────────────────────────────────────────────────────
# Speaker similarity analysis (✅ FIX 3: actually called now)
# ─────────────────────────────────────────────────────────────

def compute_speaker_similarities(real_dir, cloned_dir):
    enc = SpeakerEncoder(device="cpu")

    real_paths  = sorted(Path(real_dir).glob("*.wav"))  + sorted(Path(real_dir).glob("*.flac"))
    clone_paths = sorted(Path(cloned_dir).glob("*.wav")) + sorted(Path(cloned_dir).glob("*.flac"))

    if not real_paths or not clone_paths:
        print("[Task2] No audio files for similarity analysis.")
        return {}

    print(f"[Task2] Speaker similarity: {len(real_paths)} real + {len(clone_paths)} cloned")
    real_embs  = enc.embed_files([str(p) for p in real_paths])
    clone_embs = enc.embed_files([str(p) for p in clone_paths])

    n = min(len(real_embs), len(clone_embs))
    paired_sims = enc.cosine_similarity(real_embs[:n], clone_embs[:n])
    real_mean   = real_embs.mean(axis=0, keepdims=True)
    within_real = enc.cosine_similarity(real_embs, np.tile(real_mean, (len(real_embs), 1)))

    return {
        "real_embs":           real_embs,
        "clone_embs":          clone_embs,
        "paired_sims":         paired_sims,
        "within_real":         within_real,
        "mean_real_vs_clone":  float(np.mean(paired_sims)),
        "std_real_vs_clone":   float(np.std(paired_sims)),
        "mean_within_real":    float(np.mean(within_real)),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    real_dir   = DCFG.get("reference_speaker_dir", "./data/reference_speaker")
    cloned_dir = DCFG.get("cloned_dir",             "./data/cloned_audio")

    dataset = ClonedAudioDataset(
        real_dir=real_dir,
        cloned_dir=cloned_dir,
        sample_rate=ACFG["sample_rate"],
        max_duration_sec=ACFG["max_duration_sec"],
        n_lfcc=ACFG["n_lfcc"],
        n_fft=ACFG["n_fft"],
        hop_length=ACFG["hop_length"],
        win_length=ACFG["win_length"],
    )

    all_results = {}

    for run_id in [1]:
        model, threshold, run_name = load_model(run_id)
        labels, scores, paths = run_inference(model, dataset, run_id)

        # Score distribution plot
        real_scores  = scores[labels == 0]
        spoof_scores = scores[labels == 1]
        plot_score_distribution(
            real_scores, spoof_scores, threshold,
            title=f"{run_name} — Cloned vs Real Score Distribution",
            save_path=str(PLOT_DIR / f"{run_name}_attack_score_dist.png"),
        )

        # ✅ FIX 2: rich metrics instead of just EER
        result = compute_rich_metrics(labels, scores, threshold, run_name)

        # Load clean EER from Task 1
        clean_eer_path = f"outputs/{run_name}_results.json"
        if Path(clean_eer_path).exists():
            with open(clean_eer_path) as f:
                t1 = json.load(f)
            result["clean_eer"] = t1.get("eval_eer", 0.0)

        all_results[run_name] = result
        print_metrics_table(result, f"Task 2 — {run_name}")

    # ✅ FIX 3: speaker similarity (was commented out before)
    print("\n[Task2] Running speaker embedding similarity analysis...")
    sim_results = compute_speaker_similarities(real_dir, cloned_dir)

    if sim_results:
        print(f"  Mean cosine sim (real vs clone): {sim_results['mean_real_vs_clone']:.4f}")
        print(f"  Mean cosine sim (within real):   {sim_results['mean_within_real']:.4f}")

        plot_cosine_similarity_distribution(
            real_vs_real=sim_results["within_real"],
            real_vs_clone=sim_results["paired_sims"],
            save_path=str(PLOT_DIR / "cosine_similarity_dist.png"),
        )

        for run_name in all_results:
            all_results[run_name]["mean_cosine_sim"] = round(sim_results["mean_real_vs_clone"], 4)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/task2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Task2-FIXED] ✓ Complete.")
    print(f"  Results → outputs/task2_results.json")
    print(f"  Plots   → {PLOT_DIR}")


if __name__ == "__main__":
    main()
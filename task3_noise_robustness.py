"""scripts/task3_noise_robustness.py — Task 3: Noise stress testing.

Evaluates both classifiers under AWGN, babble, and music noise at
SNR = [0, 5, 10, 15, 20] dB. Plots bypass rate vs SNR curves.

Usage:
  python scripts/task3_noise_robustness.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcnn import LCNN
# from rawnet2 import RawNet2   # ❌ remove
from dataset import ClonedAudioDataset
from metrics import compute_bypass_rate_from_scores, print_metrics_table
from plots import plot_bypass_rate_vs_snr, plot_snr_heatmap
def add_noise(waveform, snr_db, noise_type="awgn"):
    if noise_type == "awgn":
        noise = torch.randn_like(waveform)

    elif noise_type == "babble":
        noise = torch.randn_like(waveform) * 0.5

    elif noise_type == "music":
        noise = torch.sin(torch.linspace(0, 100, waveform.shape[-1])).to(waveform.device)
        noise = noise.unsqueeze(0).unsqueeze(0).expand_as(waveform)

    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr = 10 ** (snr_db / 10)
    noise = noise * torch.sqrt(signal_power / (snr * noise_power))

    return waveform + noise

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DCFG     = cfg["data"]
ACFG     = cfg["audio"]
NCFG     = cfg["noise"]
ECFG     = cfg["evaluation"]
DEVICE   = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
PLOT_DIR = Path(ECFG["plot_dir"])
SAVE_DIR = Path(cfg["training"]["save_dir"])
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SNR_LEVELS  = NCFG["snr_levels_db"]
NOISE_TYPES = NCFG["noise_types"]


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────

def load_model(run: int):
    ckpt_path = SAVE_DIR / "lcnn_best.pt"
    model     = LCNN(input_channels=1, dropout=0.0)
    run_name  = "Run1_LCNN"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        threshold = ckpt.get("threshold", 0.5)
    else:
        print(f"[Task3] WARNING: {ckpt_path} not found.")
        threshold = 0.5

    return model.to(DEVICE).eval(), threshold, run_name

# ─────────────────────────────────────────────────────────────
# Single evaluation under noise
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_noise(
    model: torch.nn.Module,
    run: int,
    threshold: float,
    real_dir: str,
    cloned_dir: str,
    noise_type: str,
    snr_db: float,
    noise_file: str = None,
) -> dict:
    """Run inference on cloned audio under specified noise. Returns metrics dict."""

    # noise_transform = NoisyWaveformTransform(
    #     noise_type=noise_type,
    #     snr_db=snr_db,
    #     sample_rate=ACFG["sample_rate"],
    #     noise_file=noise_file,
    # )

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

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    all_labels, all_scores = [], []

    for feats, labels, _ in loader:
        feats = feats.to(DEVICE)
        feats = add_noise(feats, snr_db)
        # if run == 2:
        #     feats = feats.mean(dim=2)  # (B, 1, T)
        logits = model(feats)
        probs  = F.softmax(logits, dim=-1)[:, 1]
        all_labels.extend(labels.tolist())
        all_scores.extend(probs.cpu().tolist())

    labels_np = np.array(all_labels)
    scores_np = np.array(all_scores)

    # Bypass rate: % of cloned audio classified as real
    bypass_rate = compute_bypass_rate_from_scores(
        labels_np[labels_np == 1],
        scores_np[labels_np == 1],
        threshold=threshold,
    )

    return {
        "noise_type":  noise_type,
        "snr_db":      snr_db,
        "bypass_rate": bypass_rate,
        "n_samples":   int((labels_np == 1).sum()),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    real_dir   = DCFG.get("reference_speaker_dir", "./data/reference_speaker")
    cloned_dir = DCFG.get("cloned_dir",             "./data/cloned_audio")

    # noise file paths (optional — synthetic fallback if missing)
    babble_file = NCFG.get("babble_noise_file", "")
    music_file  = NCFG.get("music_noise_file",  "")
    noise_files = {
        "awgn":   None,
        "babble": babble_file if Path(babble_file).exists() else None,
        "music":  music_file  if Path(music_file).exists()  else None,
    }

    # results[run_name][noise_type] = [bypass_rate @ snr0, ..., bypass_rate @ snrN]
    all_results = defaultdict(lambda: defaultdict(list))
    full_log    = {}

    for run_id in [1]:
        model, threshold, run_name = load_model(run_id)
        full_log[run_name] = {}

        print(f"\n[Task3] Evaluating {run_name}...")

        for noise_type in NOISE_TYPES:
            bypass_rates = []
            full_log[run_name][noise_type] = []

            for snr_db in tqdm(SNR_LEVELS, desc=f"  {noise_type}"):
                result = evaluate_with_noise(
                    model=model,
                    run=run_id,
                    threshold=threshold,
                    real_dir=real_dir,
                    cloned_dir=cloned_dir,
                    noise_type=noise_type,
                    snr_db=snr_db,
                    noise_file=noise_files.get(noise_type),
                )
                bypass_rates.append(result["bypass_rate"])
                full_log[run_name][noise_type].append({
                    "snr_db": snr_db,
                    "bypass_rate": round(result["bypass_rate"], 4),
                })
                print(f"    {noise_type} @ {snr_db:2d}dB → bypass={result['bypass_rate']*100:.1f}%")

            all_results[run_name][noise_type] = bypass_rates

        # ── Heatmap for this run ──
        bypass_matrix = np.array([
            all_results[run_name][nt] for nt in NOISE_TYPES
        ])
        plot_snr_heatmap(
            SNR_LEVELS, NOISE_TYPES, bypass_matrix,
            run_name,
            str(PLOT_DIR / f"{run_name}_noise_heatmap.png"),
        )

    # ── Combined bypass rate vs SNR line plot ──
    plot_bypass_rate_vs_snr(
        snr_levels=SNR_LEVELS,
        results=dict(all_results),
        save_path=str(PLOT_DIR / "bypass_rate_vs_snr.png"),
    )

    # ── Summary ──
    summary = {}
    for run_name, noise_dict in all_results.items():
        summary[run_name] = {}
        for nt, rates in noise_dict.items():
            summary[run_name][nt] = {
                "bypass_at_0dB":  round(rates[SNR_LEVELS.index(0)]  * 100, 2) if 0  in SNR_LEVELS else None,
                "bypass_at_20dB": round(rates[SNR_LEVELS.index(20)] * 100, 2) if 20 in SNR_LEVELS else None,
                "mean_bypass":    round(float(np.mean(rates)) * 100, 2),
            }
        print_metrics_table(
            {f"{nt}_mean_bypass": v["mean_bypass"] for nt, v in summary[run_name].items()},
            f"Task 3 — {run_name} Summary"
        )

    # ── Save results ──
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/task3_results.json", "w") as f:
        json.dump({"summary": summary, "full_log": full_log}, f, indent=2)

    print("\n[Task3] ✓ Complete.")
    print(f"  Results → outputs/task3_results.json")
    print(f"  Plots   → {PLOT_DIR}")


if __name__ == "__main__":
    main()

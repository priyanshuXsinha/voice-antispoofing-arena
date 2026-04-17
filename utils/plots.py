"""utils/plots.py — All matplotlib plotting utilities for Tasks 1-3."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Task 1: Training curves
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: List[float],
    val_losses:   List[float],
    run_name: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#378ADD", linewidth=2)
    ax.plot(epochs, val_losses,   label="Val loss",   color="#D85A30", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(f"{run_name} — Training & Validation Loss")
    ax.legend()
    _save(fig, save_path)


def plot_eer_comparison(
    run_names: List[str],
    eers: List[float],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = ["#378ADD", "#534AB7"][:len(run_names)]
    bars = ax.bar(run_names, [e * 100 for e in eers], color=colours, width=0.4)
    ax.set_ylabel("EER (%)")
    ax.set_title("Classifier EER Comparison (lower is better)")
    for bar, eer in zip(bars, eers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{eer*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(eers) * 100 * 1.3)
    _save(fig, save_path)


def plot_score_distribution(
    real_scores: np.ndarray,
    spoof_scores: np.ndarray,
    threshold: float,
    title: str,
    save_path: str,
):
    """Histogram of spoof-scores for real vs spoof samples + EER threshold."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_scores,  bins=80, alpha=0.6, color="#1D9E75", label="Bonafide")
    ax.hist(spoof_scores, bins=80, alpha=0.6, color="#D85A30", label="Spoof")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"EER threshold ({threshold:.3f})")
    ax.set_xlabel("Spoof score P(spoof)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
# Task 2: Similarity plots
# ─────────────────────────────────────────────────────────────

def plot_cosine_similarity_distribution(
    real_vs_real: np.ndarray,
    real_vs_clone: np.ndarray,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_vs_real,  bins=60, alpha=0.65, color="#378ADD",  label="Real vs Real")
    ax.hist(real_vs_clone, bins=60, alpha=0.65, color="#BA7517",  label="Real vs Cloned")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Speaker Embedding Similarity: Real vs Cloned Audio")
    ax.legend()
    _save(fig, save_path)


def plot_attack_eer_bar(
    run_names:   List[str],
    clean_eers:  List[float],
    attack_eers: List[float],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(run_names))
    w = 0.3
    ax.bar(x - w/2, [e*100 for e in clean_eers],  w, label="Clean EER",  color="#378ADD")
    ax.bar(x + w/2, [e*100 for e in attack_eers], w, label="Attack EER", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(run_names)
    ax.set_ylabel("EER (%)")
    ax.set_title("Clean EER vs Attack EER (higher attack EER = easier to fool)")
    ax.legend()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
# Task 3: Noise robustness plots
# ─────────────────────────────────────────────────────────────

def plot_bypass_rate_vs_snr(
    snr_levels: List[float],
    results: Dict[str, Dict[str, List[float]]],
    # results[run_name][noise_type] = list of bypass rates
    save_path: str,
):
    """
    Line plot of bypass rate (%) vs SNR for all noise types and both runs.
    """
    run_colors    = {"Run1_LCNN": "#378ADD", "Run2_RawNet2": "#534AB7"}
    noise_markers = {"awgn": "o", "babble": "s", "music": "^"}
    noise_styles  = {"awgn": "-", "babble": "--", "music": "-."}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (run_name, noise_dict) in zip(axes, results.items()):
        color = run_colors.get(run_name, "#888")
        for noise_type, bypass_rates in noise_dict.items():
            ax.plot(
                snr_levels,
                [r * 100 for r in bypass_rates],
                linestyle=noise_styles.get(noise_type, "-"),
                marker=noise_markers.get(noise_type, "o"),
                color=color,
                label=noise_type.upper(),
                linewidth=2,
                markersize=6,
            )
        ax.set_title(run_name)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Bypass Rate (%)")
        ax.legend()
        ax.invert_xaxis()   # high SNR = clean on right, noisy on left

    fig.suptitle("Bypass Rate vs SNR (higher bypass = easier to fool under noise)", y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


def plot_snr_heatmap(
    snr_levels:  List[float],
    noise_types: List[str],
    bypass_matrix: np.ndarray,   # (n_noise_types, n_snr_levels)
    run_name: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        bypass_matrix * 100,
        annot=True, fmt=".1f",
        xticklabels=[f"{s}dB" for s in snr_levels],
        yticklabels=[n.upper() for n in noise_types],
        cmap="RdYlGn_r",
        vmin=0, vmax=100,
        ax=ax,
        cbar_kws={"label": "Bypass Rate (%)"},
    )
    ax.set_title(f"{run_name} — Bypass Rate Heatmap")
    ax.set_xlabel("SNR")
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
# Waveform visualisation (used in demo)
# ─────────────────────────────────────────────────────────────

def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    save_path: Optional[str] = None,
) -> plt.Figure:
    t = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(t, waveform, color="#378ADD", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    import librosa
    import librosa.display
    S = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(9, 3))
    img = librosa.display.specshow(
        S_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="hz", ax=ax, cmap="viridis"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    if save_path:
        _save(fig, save_path)
    return fig

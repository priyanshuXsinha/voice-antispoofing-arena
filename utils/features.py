"""utils/features.py — LFCC and mel-spectrogram feature extractors."""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np


# ─────────────────────────────────────────────────────────────
# LFCC (Linear Frequency Cepstral Coefficients)
# ─────────────────────────────────────────────────────────────

def extract_lfcc(
    waveform: torch.Tensor,       # shape: (T,)
    sample_rate: int = 16000,
    n_lfcc: int = 60,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """
    Extract LFCC features from a raw waveform.
    Returns tensor of shape (n_lfcc, time_frames).
    """
    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
        },
    )
    feat = lfcc_transform(waveform)  # (n_lfcc, T)

    # Apply delta and delta-delta for richer representation
    delta  = torchaudio.functional.compute_deltas(feat)
    ddelta = torchaudio.functional.compute_deltas(delta)

    # Concatenate along feature axis → (3*n_lfcc, T)
    feat = torch.cat([feat, delta, ddelta], dim=0)

    # Per-feature normalisation (mean-var)
    mean = feat.mean(dim=-1, keepdim=True)
    std  = feat.std(dim=-1, keepdim=True) + 1e-8
    feat = (feat - mean) / std

    return feat  # (3*n_lfcc, T)


# ─────────────────────────────────────────────────────────────
# Log Mel-Spectrogram
# ─────────────────────────────────────────────────────────────

def extract_mel(
    waveform: torch.Tensor,       # shape: (T,)
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 20.0,
    fmax: float = 7600.0,
) -> torch.Tensor:
    """
    Extract log-mel spectrogram from a raw waveform.
    Returns tensor of shape (n_mels, time_frames).
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
    )
    mel = mel_transform(waveform)     # (n_mels, T)
    log_mel = torch.log(mel + 1e-6)  # log-compression

    # Normalise
    mean = log_mel.mean(dim=-1, keepdim=True)
    std  = log_mel.std(dim=-1, keepdim=True) + 1e-8
    log_mel = (log_mel - mean) / std

    return log_mel  # (n_mels, T)


# ─────────────────────────────────────────────────────────────
# Waveform utilities
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(
    path: str,
    sample_rate: int = 16000,
    max_duration_sec: float = 4.0,
) -> torch.Tensor:
    """Load a .wav/.flac file, resample, mono-mix, normalise, pad/trim."""
    import torchaudio
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    max_amp = waveform.abs().max()
    if max_amp > 0:
        waveform = waveform / max_amp
    max_samples = int(max_duration_sec * sample_rate)
    T = waveform.shape[-1]
    if T >= max_samples:
        waveform = waveform[..., :max_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - T))
    return waveform  # (1, T)

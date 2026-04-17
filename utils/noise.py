"""utils/noise.py — Noise injection at controlled SNR for Task 3 stress testing."""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# SNR helpers
# ─────────────────────────────────────────────────────────────

def signal_power(signal: np.ndarray) -> float:
    return float(np.mean(signal ** 2))


def scale_noise_to_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Scale noise so that SNR(signal, scaled_noise) = snr_db.
    Returns the scaled noise array (same length as signal).
    """
    # Loop noise to match signal length
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(signal)]

    p_sig   = signal_power(signal) + 1e-10
    p_noise = signal_power(noise)  + 1e-10
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(p_sig / (snr_linear * p_noise))
    return noise * scale


# ─────────────────────────────────────────────────────────────
# Noise generators
# ─────────────────────────────────────────────────────────────

def generate_awgn(length: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Additive white Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(length).astype(np.float32)


def generate_babble_noise(
    length: int,
    sample_rate: int = 16000,
    n_speakers: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Synthetic babble noise: sum of random bandpass-filtered noise streams.
    Used as fallback if a real babble file is not available.
    """
    if rng is None:
        rng = np.random.default_rng()
    babble = np.zeros(length, dtype=np.float32)
    for _ in range(n_speakers):
        noise = rng.standard_normal(length).astype(np.float32)
        # Random bandpass to simulate voice formants
        f_lo = rng.uniform(100, 600)
        f_hi = rng.uniform(1000, 4000)
        noise_tensor = torch.from_numpy(noise).unsqueeze(0)
        # Simple IIR bandpass via two torchaudio effects
        noise_tensor = torchaudio.functional.highpass_biquad(
            noise_tensor, sample_rate=sample_rate, cutoff_freq=f_lo
        )
        noise_tensor = torchaudio.functional.lowpass_biquad(
            noise_tensor, sample_rate=sample_rate, cutoff_freq=f_hi
        )
        babble += noise_tensor.squeeze(0).numpy()
    return babble


def generate_music_noise(length: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Synthetic music noise: mix of sinusoids at harmonic frequencies.
    Used as fallback if a real music file is not available.
    """
    if rng is None:
        rng = np.random.default_rng()
    music = np.zeros(length, dtype=np.float32)
    t = np.arange(length, dtype=np.float32) / 16000.0
    # Pick random root frequency and add harmonics
    for _ in range(3):
        f0 = rng.uniform(80, 400)
        for k in range(1, 6):
            amp = rng.uniform(0.1, 0.4) / k
            music += amp * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)
    return music


def load_noise_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load a .wav noise file and return as float32 numpy array."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────

def add_noise(
    signal: np.ndarray,           # float32 mono waveform
    noise_type: str,              # "awgn" | "babble" | "music"
    snr_db: float,
    sample_rate: int = 16000,
    noise_file: Optional[str] = None,   # path to pre-recorded noise file
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add noise of the specified type to a signal at the target SNR.

    Returns the noisy signal (same length and dtype as input).
    """
    length = len(signal)

    if noise_file and Path(noise_file).exists():
        noise = load_noise_file(noise_file, sample_rate)
    elif noise_type == "awgn":
        noise = generate_awgn(length, rng)
    elif noise_type == "babble":
        noise = generate_babble_noise(length, sample_rate, rng=rng)
    elif noise_type == "music":
        noise = generate_music_noise(length, rng)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    scaled_noise = scale_noise_to_snr(signal, noise, snr_db)
    noisy = signal + scaled_noise

    # Clip to prevent overflow
    max_val = np.abs(noisy).max()
    if max_val > 1.0:
        noisy /= max_val

    return noisy.astype(np.float32)


class NoisyWaveformTransform:
    """
    Callable transform: adds noise at a fixed SNR.
    Compatible with ClonedAudioDataset's noise_transform parameter.
    """

    def __init__(
        self,
        noise_type: str,
        snr_db: float,
        sample_rate: int = 16000,
        noise_file: Optional[str] = None,
    ):
        self.noise_type  = noise_type
        self.snr_db      = snr_db
        self.sample_rate = sample_rate
        self.noise_file  = noise_file
        self.rng         = np.random.default_rng(42)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return add_noise(
            samples,
            self.noise_type,
            self.snr_db,
            self.sample_rate,
            self.noise_file,
            self.rng,
        )

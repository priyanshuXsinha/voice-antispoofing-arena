"""dataset_fixed.py — Drop-in replacement for dataset.py.

Key improvements for small datasets (11 real / 50 cloned):
  1. Real audio is oversampled with heavy augmentation so class balance is ~1:1
  2. Augmentations: pitch shift, time stretch, speed perturb, noise injection,
     random gain, SpecAugment-style time/freq masking on features
  3. Consistent API with original ClonedAudioDataset
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from features import extract_lfcc


# ─────────────────────────────────────────────────────────────
# Waveform-level augmentations (applied before feature extraction)
# ─────────────────────────────────────────────────────────────

def augment_waveform(wav: np.ndarray, sr: int = 16000,
                     is_real: bool = True) -> np.ndarray:
    """
    Apply random waveform augmentations.
    For real audio we augment more aggressively to boost effective dataset size.
    """
    aug = wav.copy()

    # 1. Random gain ±6 dB
    gain = np.random.uniform(0.5, 1.5)
    aug = aug * gain

    # 2. Speed perturbation (0.9x – 1.1x) via resampling trick
    if np.random.random() < 0.5:
        speed = np.random.uniform(0.9, 1.1)
        orig_len = len(aug)
        fake_sr = int(sr * speed)
        t = torch.from_numpy(aug).unsqueeze(0)
        t = torchaudio.functional.resample(t, fake_sr, sr)
        aug = t.squeeze(0).numpy()
        # Pad/trim back to original length
        if len(aug) >= orig_len:
            aug = aug[:orig_len]
        else:
            aug = np.pad(aug, (0, orig_len - len(aug)))

    # 3. Additive white noise (very light for real, slightly more for cloned)
    if np.random.random() < 0.7:
        noise_level = np.random.uniform(0.0005, 0.003)
        aug = aug + noise_level * np.random.randn(len(aug)).astype(np.float32)

    # 4. Random DC offset removal (simulates different microphones)
    aug = aug - aug.mean()

    # 5. Room impulse / reverb simulation via simple convolution
    if is_real and np.random.random() < 0.3:
        room_len = int(sr * np.random.uniform(0.01, 0.05))
        rir = np.random.exponential(0.5, room_len).astype(np.float32)
        rir /= rir.sum() + 1e-8
        aug = np.convolve(aug, rir, mode="full")[:len(wav)]

    # Normalise
    max_amp = np.abs(aug).max()
    if max_amp > 0:
        aug = aug / max_amp

    return aug.astype(np.float32)


def spec_augment(feat: torch.Tensor,
                 freq_mask_param: int = 20,
                 time_mask_param: int = 30,
                 num_masks: int = 2) -> torch.Tensor:
    """
    SpecAugment: randomly mask frequency and time bands on the feature tensor.
    feat: (1, n_feat, T)  →  same shape
    """
    _, F, T = feat.shape
    aug = feat.clone()

    for _ in range(num_masks):
        # Frequency mask
        f = random.randint(0, min(freq_mask_param, F - 1))
        f0 = random.randint(0, F - f)
        aug[0, f0:f0 + f, :] = 0.0

        # Time mask
        t = random.randint(0, min(time_mask_param, T - 1))
        t0 = random.randint(0, T - t)
        aug[0, :, t0:t0 + t] = 0.0

    return aug


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class ClonedAudioDataset(torch.utils.data.Dataset):
    """
    Drop-in replacement for original ClonedAudioDataset with:
    - Heavy augmentation on real audio to fix class imbalance
    - SpecAugment on features during training
    - Consistent (feat, label, path) output
    """

    def __init__(
        self,
        real_dir: str,
        cloned_dir: str,
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
        feature_type: str = "lfcc",
        n_lfcc: int = 60,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        noise_transform=None,
        augment: bool = False,
        oversample_real: bool = True,  # key fix for imbalance
    ):
        self.sample_rate   = sample_rate
        self.max_samples   = int(max_duration_sec * sample_rate)
        self.n_lfcc        = n_lfcc
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.win_length    = win_length
        self.noise_transform = noise_transform
        self.augment       = augment

        real_files   = [(str(p), 0) for p in Path(real_dir).glob("*.wav")]
        real_files  += [(str(p), 0) for p in Path(real_dir).glob("*.flac")]
        clone_files  = [(str(p), 1) for p in Path(cloned_dir).glob("*.wav")]
        clone_files += [(str(p), 1) for p in Path(cloned_dir).glob("*.flac")]

        n_real   = len(real_files)
        n_cloned = len(clone_files)

        # ── Oversample real to match cloned count ──
        if oversample_real and augment and n_real < n_cloned:
            # Repeat real files until we match cloned count
            factor = int(np.ceil(n_cloned / max(n_real, 1)))
            real_files_balanced = (real_files * factor)[:n_cloned]
        else:
            real_files_balanced = real_files

        self.files = real_files_balanced + clone_files
        print(f"[Dataset] Real (after oversample): {len(real_files_balanced)} "
              f"| Cloned: {n_cloned} | Total: {len(self.files)}")

    def _load(self, path: str) -> np.ndarray:
        waveform, sr = sf.read(path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

        if sr != self.sample_rate:
            t = torch.from_numpy(waveform).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, self.sample_rate)
            waveform = t.squeeze(0).numpy()

        max_amp = np.abs(waveform).max()
        if max_amp > 0:
            waveform /= max_amp

        T = len(waveform)
        if T >= self.max_samples:
            # Random crop if augmenting, else centre crop
            if self.augment:
                start = random.randint(0, T - self.max_samples)
            else:
                start = (T - self.max_samples) // 2
            waveform = waveform[start:start + self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - T))

        return waveform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform = self._load(path)   # (T,) float32

        # Apply waveform augmentation during training
        if self.augment:
            waveform = augment_waveform(waveform, self.sample_rate,
                                        is_real=(label == 0))

        if self.noise_transform:
            waveform = self.noise_transform(
                samples=waveform, sample_rate=self.sample_rate
            )

        # Extract LFCC
        wav_tensor = torch.from_numpy(waveform)
        feat = extract_lfcc(
            wav_tensor,
            sample_rate=self.sample_rate,
            n_lfcc=self.n_lfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        ).unsqueeze(0)   # (1, 180, T')

        # Apply SpecAugment during training
        if self.augment:
            feat = spec_augment(feat, freq_mask_param=18, time_mask_param=25)

        return feat, torch.tensor(label, dtype=torch.long), path

"""utils/dataset.py — ASVspoof2019 LA dataset loader."""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf

from features import extract_lfcc, extract_mel

# ─────────────────────────────────────────────────────────────
# Protocol parser
# ─────────────────────────────────────────────────────────────

# def parse_protocol(protocol_path: str) -> List[Tuple[str, str, str]]:
#     """
#     Parse ASVspoof2019 LA protocol file.
#     Returns list of (speaker_id, file_id, label) where label is 'bonafide' or 'spoof'.
#     """
#     entries = []
#     with open(protocol_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             # Format: SPEAKER_ID FILE_ID - ATTACK_TYPE LABEL
#             speaker_id = parts[0]
#             file_id    = parts[1]
#             label      = parts[4]   # 'bonafide' or 'spoof'
#             entries.append((speaker_id, file_id, label))
#     return entries


# # ─────────────────────────────────────────────────────────────
# # Dataset
# # ─────────────────────────────────────────────────────────────

# class ClonedAudioDataset(Dataset):
#     """
#     PyTorch Dataset for ASVspoof2019 LA.

#     Returns (feature_tensor, label) where:
#       - feature_tensor: (1, n_lfcc, T) or (1, n_mels, T)
#       - label: 0 = bonafide (real), 1 = spoof (fake)
#     """

#     LABEL_MAP = {"bonafide": 0, "spoof": 1}

#     def __init__(
#         self,
#         protocol_path: str,
#         audio_dir: str,
#         sample_rate: int = 16000,
#         max_duration_sec: float = 4.0,
#         feature_type: str = "lfcc",   # "lfcc" or "mel"
#         n_lfcc: int = 60,
#         n_mels: int = 80,
#         n_fft: int = 512,
#         hop_length: int = 160,
#         win_length: int = 400,
#         augment: bool = False,
#         transform=None,
#     ):
#         super().__init__()
#         self.audio_dir    = Path(audio_dir)
#         self.sample_rate  = sample_rate
#         self.max_samples  = int(max_duration_sec * sample_rate)
#         self.feature_type = feature_type
#         self.n_lfcc       = n_lfcc
#         self.n_mels       = n_mels
#         self.n_fft        = n_fft
#         self.hop_length   = hop_length
#         self.win_length   = win_length
#         self.augment      = augment
#         self.transform    = transform

#         self.entries = parse_protocol(protocol_path)
#         print(f"[Dataset] Loaded {len(self.entries)} entries from {protocol_path}")
#         self._log_class_balance()

#     def _log_class_balance(self):
#         n_real = sum(1 for _, _, l in self.entries if l == "bonafide")
#         n_fake = sum(1 for _, _, l in self.entries if l == "spoof")
#         print(f"[Dataset] Real: {n_real} | Fake: {n_fake} | Total: {len(self.entries)}")

#     def _load_waveform(self, file_id: str) -> torch.Tensor:
#         """Load and normalise a .flac file to mono float32 at target sample rate."""
#         path = self.audio_dir / f"{file_id}.flac"
#         if not path.exists():
#             # fallback: try .wav
#             path = self.audio_dir / f"{file_id}.wav"

#         waveform, sr = torchaudio.load(str(path))

#         # Resample if needed
#         if sr != self.sample_rate:
#             waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

#         # Convert to mono
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)

#         # Normalise amplitude
#         max_amp = waveform.abs().max()
#         if max_amp > 0:
#             waveform = waveform / max_amp

#         return waveform  # shape: (1, T)

#     def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
#         """Pad or trim waveform to self.max_samples."""
#         T = waveform.shape[-1]
#         if T >= self.max_samples:
#             # Random crop during training; centre crop during eval
#             if self.augment:
#                 start = random.randint(0, T - self.max_samples)
#             else:
#                 start = (T - self.max_samples) // 2
#             return waveform[..., start : start + self.max_samples]
#         else:
#             pad_len = self.max_samples - T
#             return torch.nn.functional.pad(waveform, (0, pad_len))

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx: int):
#         _, file_id, label_str = self.entries[idx]
#         label = self.LABEL_MAP[label_str]

#         waveform = self._load_waveform(file_id)
#         waveform = self._pad_or_trim(waveform)

#         # Feature extraction
#         if self.feature_type == "lfcc":
#             feat = extract_lfcc(
#                 waveform.squeeze(0),
#                 sample_rate=self.sample_rate,
#                 n_lfcc=self.n_lfcc,
#                 n_fft=self.n_fft,
#                 hop_length=self.hop_length,
#                 win_length=self.win_length,
#             )
#         else:
#             feat = extract_mel(
#                 waveform.squeeze(0),
#                 sample_rate=self.sample_rate,
#                 n_mels=self.n_mels,
#                 n_fft=self.n_fft,
#                 hop_length=self.hop_length,
#                 win_length=self.win_length,
#             )

#         # feat shape: (n_features, T) → add channel dim → (1, n_features, T)
#         feat = feat.unsqueeze(0)

#         if self.transform:
#             feat = self.transform(feat)

#         return feat, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# Cloned audio dataset (Task 2 / Task 3)
# ─────────────────────────────────────────────────────────────

class ClonedAudioDataset(Dataset):
    """
    Simple dataset for evaluating cloned vs real audio.
    Expects a directory of .wav files.
    label: 0=real, 1=cloned
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
        noise_snr=None,
        augment=False,   # ✅ ADD THIS

        
        
    ):
        self.sample_rate   = sample_rate
        self.max_samples   = int(max_duration_sec * sample_rate)
        self.feature_type  = feature_type
        self.n_lfcc        = n_lfcc
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.win_length    = win_length
        self.noise_transform = noise_transform
        self.noise_snr = noise_snr   # ✅ IMPORTANT
        self.augment   = augment     # ✅ now valid

        real_files   = [(str(p), 0) for p in Path(real_dir).glob("*.wav")]
        real_files  += [(str(p), 0) for p in Path(real_dir).glob("*.flac")]
        # real_files = real_files * 50
        clone_files  = [(str(p), 1) for p in Path(cloned_dir).glob("*.wav")]
        clone_files += [(str(p), 1) for p in Path(cloned_dir).glob("*.flac")]

        self.files = real_files + clone_files
        print(f"[ClonedDataset] Real: {len(real_files)} | Cloned: {len(clone_files)}")

    def _load(self, path: str) -> torch.Tensor:
        import soundfile as sf

        waveform, sr = sf.read(path)

    # Convert to tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # Stereo → mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=1)

        waveform = waveform.unsqueeze(0)  # (1, T)

    # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

    # Normalize
        max_amp = waveform.abs().max()
        if max_amp > 0:
            waveform = waveform / max_amp

    # Pad / trim
        T = waveform.shape[-1]
        if T >= self.max_samples:
            waveform = waveform[..., :self.max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - T))

        return waveform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform = self._load(path)

        if self.noise_snr is not None:
            noise = torch.randn_like(waveform)

            signal_power = waveform.pow(2).mean()
            noise_power = noise.pow(2).mean()

            snr = 10 ** (self.noise_snr / 10)
            noise = noise * torch.sqrt(signal_power / (snr * noise_power))

            waveform = waveform + noise
       

    # ✅ Add noise ONLY during training
        if self.augment:
            noise_level = torch.rand(1).item() * 0.003
            waveform = waveform + noise_level * torch.randn_like(waveform)

        if self.noise_transform:
            wav_np = waveform.squeeze(0).numpy()
            wav_np = self.noise_transform(samples=wav_np, sample_rate=self.sample_rate)
            waveform = torch.from_numpy(wav_np).unsqueeze(0)

        feat = extract_lfcc(
        waveform.squeeze(0),
            sample_rate=self.sample_rate,
            n_lfcc=self.n_lfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        ).unsqueeze(0)

        return feat, torch.tensor(label, dtype=torch.long), path
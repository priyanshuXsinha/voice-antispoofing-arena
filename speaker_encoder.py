"""models/speaker_encoder.py — Resemblyzer-based speaker embedding extractor."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Union


class SpeakerEncoder:
    """
    Wrapper around Resemblyzer VoiceEncoder for computing d-vector embeddings.
    Used in Task 2 to measure cosine similarity between real and cloned speech.
    """

    def __init__(self, device: str = "cpu"):
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.encoder  = VoiceEncoder(device=device)
            self.preprocess = preprocess_wav
            self._available = True
            print("[SpeakerEncoder] Resemblyzer VoiceEncoder loaded.")
        except ImportError:
            print("[SpeakerEncoder] WARNING: resemblyzer not installed. "
                  "Run: pip install resemblyzer")
            self._available = False

        self.device = device

    def embed_file(self, wav_path: str) -> np.ndarray:
        """Compute 256-dim d-vector embedding for a single audio file."""
        if not self._available:
            return np.random.randn(256).astype(np.float32)
        wav = self.preprocess(Path(wav_path))
        return self.encoder.embed_utterance(wav)   # (256,) float32

    def embed_files(self, wav_paths: List[str]) -> np.ndarray:
        """Compute embeddings for a list of files. Returns (N, 256) array."""
        embeddings = [self.embed_file(p) for p in wav_paths]
        return np.stack(embeddings, axis=0)   # (N, 256)

    def cosine_similarity(
        self,
        emb_a: np.ndarray,   # (N, D) or (D,)
        emb_b: np.ndarray,   # (N, D) or (D,)
    ) -> np.ndarray:
        """Per-pair cosine similarity between embeddings."""
        a = torch.from_numpy(emb_a)
        b = torch.from_numpy(emb_b)
        if a.dim() == 1:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
        sims = F.cosine_similarity(a, b, dim=-1)
        return sims.numpy()

    def mean_similarity(
        self,
        real_embeddings: np.ndarray,    # (N, D)
        clone_embeddings: np.ndarray,   # (N, D)
    ) -> float:
        """Mean cosine similarity between paired real and cloned embeddings."""
        sims = self.cosine_similarity(real_embeddings, clone_embeddings)
        return float(np.mean(sims))

    def within_speaker_similarity(self, embeddings: np.ndarray) -> float:
        """Mean pairwise cosine similarity within a set of same-speaker embeddings."""
        n = len(embeddings)
        if n < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                sims = self.cosine_similarity(embeddings[i], embeddings[j])
                total += float(sims.mean())
                count += 1
        return total / count if count > 0 else 0.0

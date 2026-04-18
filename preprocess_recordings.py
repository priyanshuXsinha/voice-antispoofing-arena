"""preprocess_recordings_fixed.py — Accent-preserving preprocessing for Indian English.

Key fixes over original:
  1. Gentle denoising (alpha=0.8 instead of 2.0) — preserves formants that carry your accent
  2. Wiener-style spectral flooring instead of hard subtraction
  3. Builds a reference WAV without silence gaps (XTTS-v2 prefers continuous speech)
  4. Verifies minimum reference duration (XTTS needs ≥10s, ideally 20–30s)
  5. Exports a "raw" reference too (no denoising) — use this if XTTS still sounds wrong

Usage:
  python preprocess_recordings_fixed.py
  python preprocess_recordings_fixed.py --keep 8 --no_denoise   # skip denoising entirely
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 16000


# ─────────────────────────────────────────────────────────────
# Audio cleaning
# ─────────────────────────────────────────────────────────────

def load_mono_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        wav = t.squeeze(0).numpy()
    return wav


def trim_silence(wav: np.ndarray, threshold: float = 0.008,
                 frame_ms: int = 20) -> np.ndarray:
    """Remove leading and trailing silence. Lower threshold = keep more speech."""
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    energy = np.array([
        np.sqrt(np.mean(wav[i:i + frame_len] ** 2))
        for i in range(0, len(wav) - frame_len, frame_len)
    ])
    active = np.where(energy > threshold)[0]
    if len(active) == 0:
        return wav
    start = active[0] * frame_len
    end = (active[-1] + 1) * frame_len
    return wav[start:end]


def spectral_denoise_gentle(wav: np.ndarray, noise_frames: int = 15,
                             alpha: float = 0.8) -> np.ndarray:
    """
    GENTLE spectral subtraction — removes hiss but PRESERVES vocal character.

    Critical change from original:
      alpha = 0.8  (was 2.0 — that was destroying your accent-carrying formants)

    The Wiener-style floor (min=0.6*magnitude) ensures we never blank out
    frequency bands that contain your vowels and consonant transitions.
    """
    n_fft = 512
    hop = 128
    frame_len = n_fft

    t = torch.from_numpy(wav).unsqueeze(0)
    window = torch.hann_window(frame_len)
    spec = torch.stft(t.squeeze(0), n_fft=n_fft, hop_length=hop,
                      win_length=frame_len, window=window,
                      return_complex=True)

    magnitude = spec.abs()
    phase = spec.angle()

    # Use a short noise estimate (first ~300ms of recording = pre-speech noise)
    noise_est = magnitude[:, :noise_frames].mean(dim=1, keepdim=True)

    # Gentle subtraction: floor at 60% of original (not 10%)
    # This keeps vocal texture while reducing steady-state noise
    cleaned = torch.clamp(
        magnitude - alpha * noise_est,
        min=0.6 * magnitude          # <-- KEY FIX: was 0.1, now 0.6
    )

    cleaned_complex = torch.polar(cleaned, phase)
    out = torch.istft(cleaned_complex, n_fft=n_fft, hop_length=hop,
                      win_length=frame_len, window=window,
                      length=len(wav))

    return out.numpy().astype(np.float32)


def normalise_loudness(wav: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-8
    return (wav * target_rms / rms).clip(-1.0, 1.0)


def estimate_snr(wav: np.ndarray, noise_frames: int = 15) -> float:
    frame = int(SAMPLE_RATE * 0.02)
    noise_len = noise_frames * frame
    if len(wav) < noise_len * 2:
        return 0.0
    noise_power = np.mean(wav[:noise_len] ** 2) + 1e-10
    signal_power = np.mean(wav[noise_len:] ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)


def clean_recording(path: str, denoise: bool = True) -> tuple:
    wav = load_mono_16k(path)
    snr_before = estimate_snr(wav)
    if denoise:
        wav = spectral_denoise_gentle(wav)   # gentle, not aggressive
    wav = trim_silence(wav, threshold=0.008)  # lower threshold = keep more of your speech
    wav = normalise_loudness(wav)
    snr_after = estimate_snr(wav)
    return wav, snr_before, snr_after


# ─────────────────────────────────────────────────────────────
# Reference WAV builder — NO silence gaps between clips
# ─────────────────────────────────────────────────────────────

def build_reference_wav(recordings: list, out_path: str,
                        max_sec: int = 30) -> float:
    """
    Concatenate cleaned recordings with a VERY SHORT crossfade (no dead silence).
    XTTS-v2's speaker encoder works better with continuous natural speech
    than with silence-padded chunks.

    Returns total duration in seconds.
    """
    parts = []
    crossfade_len = int(SAMPLE_RATE * 0.05)  # 50ms crossfade, not 300ms silence

    for i, (_, wav, *_) in enumerate(recordings):
        if i == 0:
            parts.append(wav)
        else:
            prev = parts[-1]
            # Simple linear crossfade at the boundary
            fade_out = np.linspace(1, 0, crossfade_len)
            fade_in = np.linspace(0, 1, crossfade_len)
            if len(prev) >= crossfade_len and len(wav) >= crossfade_len:
                prev[-crossfade_len:] *= fade_out
                wav_copy = wav.copy()
                wav_copy[:crossfade_len] *= fade_in
                parts.append(wav_copy)
            else:
                parts.append(wav)

    combined = np.concatenate(parts)
    max_len = SAMPLE_RATE * max_sec
    if len(combined) > max_len:
        combined = combined[:max_len]

    sf.write(out_path, combined, SAMPLE_RATE)
    return len(combined) / SAMPLE_RATE


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="data/reference_speaker")
    parser.add_argument("--output_dir", default="data/reference_speaker_clean")
    parser.add_argument("--keep", type=int, default=None)
    parser.add_argument("--no_denoise", action="store_true",
                        help="Skip denoising entirely (try this if voice still sounds wrong)")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.wav")) + sorted(in_dir.glob("*.flac"))
    if not files:
        print(f"No audio files found in {in_dir}")
        return

    denoise = not args.no_denoise
    print(f"Processing {len(files)} recordings | denoising={'gentle' if denoise else 'OFF'}\n")

    results = []
    for path in files:
        try:
            wav, snr_before, snr_after = clean_recording(str(path), denoise=denoise)
            duration = len(wav) / SAMPLE_RATE
            results.append((path.name, wav, snr_before, snr_after, duration))
            print(f"  {path.name:25s}  duration={duration:.1f}s  "
                  f"SNR: {snr_before:.1f}dB → {snr_after:.1f}dB")
        except Exception as e:
            print(f"  ✗ {path.name}: {e}")

    # Sort by SNR — best recordings first
    results.sort(key=lambda x: x[3], reverse=True)

    if args.keep:
        results = results[:args.keep]
        print(f"\n[Keeping top {len(results)} recordings by SNR]")

    # Save individual cleaned files
    print(f"\nSaving {len(results)} cleaned files to {out_dir} ...")
    for i, (name, wav, _, snr_after, duration) in enumerate(results):
        out_name = f"speaker{i+1:02d}_clean.wav"
        sf.write(str(out_dir / out_name), wav, SAMPLE_RATE)
        print(f"  {name} → {out_name}  (SNR={snr_after:.1f}dB, {duration:.1f}s)")

    # ── Build combined reference (with crossfade, no dead silence) ──
    ref_path = str(out_dir / "_reference_combined.wav")
    total_dur = build_reference_wav(results, ref_path, max_sec=30)

    print(f"\n✓ Combined reference → {ref_path}  ({total_dur:.1f}s)")

    # ── Duration warning ──
    if total_dur < 10:
        print("\n⚠  WARNING: Reference is only {:.1f}s — XTTS-v2 needs ≥10s for good accent transfer.".format(total_dur))
        print("   Record more clips or use --no_denoise to avoid trimming too much silence.")
    elif total_dur < 20:
        print(f"\n⚠  Reference is {total_dur:.1f}s — works but 20–30s gives much better accent match.")
    else:
        print(f"\n✓  Reference duration {total_dur:.1f}s — good for accent transfer.")

    # ── Also save a raw (no denoising) reference for comparison ──
    raw_results = []
    for path in files[:len(results)]:
        try:
            wav = load_mono_16k(str(path))
            wav = trim_silence(wav, threshold=0.008)
            wav = normalise_loudness(wav)
            raw_results.append((path.name, wav, 0, 0, len(wav)/SAMPLE_RATE))
        except Exception:
            pass
    if raw_results:
        raw_ref_path = str(out_dir / "_reference_raw.wav")
        build_reference_wav(raw_results, raw_ref_path, max_sec=30)
        print(f"✓  Raw reference (no denoising) → {raw_ref_path}")
        print("   If XTTS still sounds wrong, try --ref pointing to the raw reference.")

    print(f"""
Next steps:
  # Try denoised reference first (recommended):
  python task0_voice_clone_fixed2.py --ref {ref_path} --tts xtts --n 60

  # If voice doesn't match, try raw reference (no denoising):
  python task0_voice_clone_fixed2.py --ref {out_dir}/_reference_raw.wav --tts xtts --n 60
""")


if __name__ == "__main__":
    main()
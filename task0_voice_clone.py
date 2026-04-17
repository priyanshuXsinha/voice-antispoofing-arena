"""scripts/task0_voice_clone.py — Task 0: Voice cloning using Coqui TTS (YourTTS).

Steps:
  1. Load reference speaker audio from data/reference_speaker/
  2. Use YourTTS (multilingual, zero-shot speaker cloning) to synthesise
     a set of utterances in the target speaker's voice.
  3. Save paired (real, cloned) manifest as CSV.

Usage:
  python scripts/task0_voice_clone.py
"""

import os
import sys
import csv
import glob
import random
from pathlib import Path

import yaml
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

VC = cfg["voice_clone"]
AUDIO = cfg["audio"]
OUTPUT_DIR = Path(VC["output_dir"])
REF_DIR    = Path(cfg["data"]["reference_speaker_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Sentences to synthesise (loaded from file or built-in)
# ─────────────────────────────────────────────────────────────

BUILTIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "Peter Piper picked a peck of pickled peppers.",
    "All that glitters is not gold.",
    "To be or not to be, that is the question.",
    "I am speaking in my own voice right now.",
    "The weather today is quite pleasant and warm.",
    "Artificial intelligence is transforming many industries.",
    "Voice synthesis technology has advanced remarkably.",
    "The sun rises in the east and sets in the west.",
    "We are testing a voice cloning system today.",
    "Security researchers use these tools to study spoofing.",
    "The cat sat on the mat and looked around curiously.",
    "She opened the door and stepped into the bright sunlight.",
    "He walked quickly across the empty parking lot.",
    "The train departed at six in the morning.",
    "I think therefore I am, as the philosopher said.",
    "Computing machines are becoming smarter every year.",
    "Please verify your identity before proceeding.",
    "The digital revolution changed how we communicate.",
    "Biometric systems can be vulnerable to synthetic speech.",
    "Audio signals carry rich information about the speaker.",
    "Deep learning models can now generate realistic speech.",
    "Signal processing is a fundamental part of audio research.",
    "The sample rate determines the quality of the recording.",
    "Anti-spoofing classifiers detect fake speech signals.",
    "Neural networks learn complex patterns from large datasets.",
    "Voice characteristics include pitch, timbre, and rhythm.",
    "Researchers continue to improve speaker verification systems.",
    "The model was trained on thousands of hours of audio data.",
    "Feature extraction is a critical step in speech processing.",
    "Cepstral coefficients are widely used in speech analysis.",
    "Mel frequency features capture perceptual properties of sound.",
    "Gradient descent optimises the loss function iteratively.",
    "Batch normalisation helps stabilise neural network training.",
    "Dropout prevents overfitting by randomly deactivating neurons.",
    "Attention mechanisms allow models to focus on relevant parts.",
    "Transfer learning leverages knowledge from pretrained models.",
    "The evaluation set measures generalisation to unseen data.",
    "False acceptance rate and false rejection rate are key metrics.",
    "Equal error rate is where these two metrics are equal.",
    "Speaker embeddings capture the identity of the speaker.",
    "Cosine similarity measures the angle between two vectors.",
    "Audio augmentation makes models more robust to noise.",
    "White noise is a random signal with a flat power spectrum.",
    "Babble noise simulates a crowded environment with many speakers.",
    "Signal to noise ratio quantifies the clarity of a recording.",
    "Real-time inference requires low-latency model architectures.",
    "The demo allows users to test the system interactively.",
]


def load_texts(texts_file: str, n: int) -> list:
    p = Path(texts_file)
    if p.exists():
        with open(p) as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"[Task0] Loaded {len(texts)} sentences from {texts_file}")
    else:
        texts = BUILTIN_TEXTS
        print(f"[Task0] Using {len(texts)} built-in sentences.")
    random.seed(42)
    random.shuffle(texts)
    return texts[:n]


# ─────────────────────────────────────────────────────────────
# Reference audio selection
# ─────────────────────────────────────────────────────────────

def find_reference_audio(ref_dir: Path) -> str:
    """Return path to best reference audio (longest file, or first .wav)."""
    wavs = sorted(ref_dir.glob("*.wav")) + sorted(ref_dir.glob("*.flac"))
    if not wavs:
        raise FileNotFoundError(
            f"No .wav or .flac files found in {ref_dir}. "
            "Please add your reference speaker audio there."
        )
    # Pick the longest file (more speaker info = better cloning)
    best = max(wavs, key=lambda p: p.stat().st_size)
    print(f"[Task0] Using reference audio: {best}")
    return str(best)


# ─────────────────────────────────────────────────────────────
# Voice cloning
# ─────────────────────────────────────────────────────────────

def clone_with_coqui_yourtts(
    texts: list,
    reference_audio: str,
    output_dir: Path,
    sample_rate: int = 16000,
) -> list:
    try:
        from TTS.api import TTS
    except ImportError:
        print("[Task0] ERROR: TTS not installed. Run: pip install TTS")
        sys.exit(1)

    print("[Task0] Loading YourTTS model (first run will download ~1.5GB)...")
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=True,
        gpu=torch.cuda.is_available()
    )

    output_paths = []

    for i, text in enumerate(tqdm(texts, desc="Cloning utterances")):
        out_path = output_dir / f"clone_{i:04d}.wav"

        try:
            tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language="en",
            file_path=str(out_path),
        )

            print(f"✅ Saved: {out_path}")

        # ✅ ADD THIS LINE HERE (before torchaudio)
            output_paths.append(str(out_path))

        # OPTIONAL: resample (can skip completely)
            try:
                waveform, sr = torchaudio.load(str(out_path))
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                    torchaudio.save(str(out_path), waveform, sample_rate)
            except Exception as e:
                print(f"⚠️ Resample skipped: {e}")

        except Exception as e:
            print(f"❌ Failed {i}: {e}")
    return output_paths

# ─────────────────────────────────────────────────────────────
# Manifest writer
# ─────────────────────────────────────────────────────────────

def write_manifest(
    real_paths: list,
    cloned_paths: list,
    output_path: str,
):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "path", "text"])
        for p in real_paths:
            writer.writerow(["real",   p, ""])
        for p in cloned_paths:
            i = int(Path(p).stem.split("_")[-1])
            writer.writerow(["cloned", p, ""])
    print(f"[Task0] Manifest saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    n_utterances = VC.get("num_utterances", 50)
    texts = load_texts(VC.get("texts_file", ""), n_utterances)

    # Reference audio
    use_cloning = True
    try:
        reference_audio = find_reference_audio(REF_DIR)
    except FileNotFoundError as e:
        print(f"[Task0] {e}")
        print("[Task0] Falling back to Tacotron2 (no speaker cloning).")
        use_cloning = False

    # Real speaker files
    real_paths = sorted(str(p) for p in REF_DIR.glob("*.wav"))
    real_paths += sorted(str(p) for p in REF_DIR.glob("*.flac"))

    # Clone
    if use_cloning:
        cloned_paths = clone_with_coqui_yourtts(
            texts, reference_audio, OUTPUT_DIR, AUDIO["sample_rate"]
        )
    else:
        cloned_paths = clone_with_tacotron2_fallback(
            texts, OUTPUT_DIR, AUDIO["sample_rate"]
        )

    print(f"\n[Task0] Generated {len(cloned_paths)} cloned utterances → {OUTPUT_DIR}")

    # Save manifest
    manifest_path = "outputs/task0_manifest.csv"
    os.makedirs("outputs", exist_ok=True)
    write_manifest(real_paths, cloned_paths, manifest_path)

    print("\n[Task0] ✓ Complete.")
    print(f"  Reference audio dir : {REF_DIR}")
    print(f"  Cloned audio dir    : {OUTPUT_DIR}")
    print(f"  Manifest            : {manifest_path}")
    print(f"  Total cloned files  : {len(cloned_paths)}")
    


if __name__ == "__main__":
    main()

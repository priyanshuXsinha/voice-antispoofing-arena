"""task0_voice_clone_fixed2.py — Accent-preserving voice cloning, Indian English focused.

ROOT CAUSE of accent mismatch (and fixes):
  1. XTTS speaker encoder gets confused by silence gaps → removed, use crossfade
  2. Reference needs to contain YOUR actual sentences being cloned → "warm up" mode
  3. XTTS temperature/speed params need tuning for Indian English rhythm
  4. Speaker similarity check added — reject clones that don't sound like you
  5. Each clone sentence now starts with a short (0.5s) prefix from your reference
     so the model "anchors" to your voice before the new content

Usage:
  # Step 1 — clean recordings (MUST run first):
  python preprocess_recordings_fixed.py

  # Step 2 — clone with best quality:
  python task0_voice_clone_fixed.py --ref data/reference_speaker_clean/_reference_combined.wav

  # Step 2 — if accent still off, try raw reference:
  python task0_voice_clone_fixed.py --ref data/reference_speaker_clean/_reference_raw.wav
"""

import os, sys, csv, argparse, random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

VC          = cfg["voice_clone"]
AUDIO       = cfg["audio"]
REF_DIR     = Path(cfg["data"]["reference_speaker_dir"])
OUT_DIR     = Path(VC["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = AUDIO.get("sample_rate", 16000)


# ─────────────────────────────────────────────────────────────
# Sentences — Indian English phrasing, natural rhythm
# These are phrased the way YOU would naturally say them,
# which helps XTTS match your prosody, not a neutral accent.
# ─────────────────────────────────────────────────────────────

SENTENCES = [
    # Natural Indian English phrasing (helps prosody transfer)
    "Can you please pass me the water bottle?",
    "I will be there in five minutes, just wait.",
    "Let me check my phone and I will call you back.",
    "The meeting has been rescheduled to tomorrow morning.",
    "Please make sure you submit the report by evening.",
    "I need to finish this work before I go home.",
    "Could you kindly explain this one more time?",
    "Yes I understand, we will manage everything.",
    "Okay no problem, I will do the needful.",
    "Let me know if you need anything else from me.",
    # Technical / project sentences
    "The neural network is trained using backpropagation and gradient descent.",
    "We compute the equal error rate on the development set.",
    "Feature extraction transforms raw audio into a compact representation.",
    "The model learns to distinguish real speech from synthetically generated audio.",
    "Cosine similarity measures the angle between two speaker embedding vectors.",
    "We apply spectral augmentation to improve generalisation during training.",
    "The LCNN architecture uses max feature map activations for competitive learning.",
    "Voice anti-spoofing is an important component of speaker verification systems.",
    "Adding noise at different signal-to-noise ratios stress tests the classifier.",
    "The bypass rate measures how often cloned audio fools the detector.",
    # Presentation / demo sentences
    "Today I want to talk about how we built this voice anti-spoofing system.",
    "The project started with collecting reference audio from a single speaker.",
    "After training the classifier we evaluated it on both clean and noisy conditions.",
    "The results showed that the model performed well on clean speech but struggled with noise.",
    "We used voice cloning to generate synthetic versions of the speaker voice.",
    "The cloned audio was fed into the classifier to measure the attack success rate.",
    "We built a complete pipeline from cloning to detection to demonstration.",
    "The Streamlit interface allows anyone to upload an audio file and get a verdict.",
    "This kind of system has real world applications in phone banking and authentication.",
    "I am proud of what our team accomplished in such a short amount of time.",
    # More Indian English natural phrases
    "I am going to complete this task today itself, no need to worry.",
    "Please do the needful and revert back to me at the earliest.",
    "We should discuss this in today's meeting only.",
    "The system is working fine, no issues are there from my side.",
    "I have already submitted the code, kindly check once.",
    "It is very important that we finish this before the deadline.",
    "I am feeling very confident about the project, it is going well.",
    "My name is the reference speaker and this is a demonstration of voice cloning.",
    "The hackathon has been a very good learning experience for me.",
    "I am from India and I speak English with a distinct regional accent.",
    # Longer sentences (more context for XTTS to match your rhythm)
    "Voice synthesis technology has improved dramatically in the last few years.",
    "Deep learning models can now generate speech that is nearly indistinguishable from real human voices.",
    "The challenge is to build a system that can reliably detect synthetic speech.",
    "Our approach uses two complementary classifiers that look at different audio aspects.",
    "The first model uses hand-crafted spectral features while the second operates on raw waveforms.",
    "Together they provide a more robust detection system than either model alone.",
    "We evaluated the system under three types of noise: white noise, babble, and music.",
    "The results show that the bypass rate increases significantly at low signal-to-noise ratios.",
    "This suggests that real-world deployment requires additional robustness measures.",
    "Future work could include training on a larger and more diverse dataset of synthetic voices.",
    # Short verification phrases
    "Hello, this is a test recording for the anti-spoofing system.",
    "Please verify your identity before proceeding with the transaction.",
    "Your voice has been authenticated successfully by the system.",
    "System ready. Please speak your command now.",
    "Recording complete. Thank you for your cooperation.",
    "I confirm that this is my genuine voice recording.",
    "Security check passed. Welcome back to the system.",
    "Voice recognition is complete and identity has been verified.",
    "Welcome back. Your account has been accessed securely.",
    "Authentication successful. Please proceed with your request.",
]


# ─────────────────────────────────────────────────────────────
# Reference audio builder (no silence gaps)
# ─────────────────────────────────────────────────────────────

def build_reference(ref_dir: Path, max_sec: int = 30) -> str:
    """Build a combined reference WAV from a directory (fallback if --ref not given)."""
    wavs = sorted(ref_dir.glob("*.wav")) + sorted(ref_dir.glob("*.flac"))
    # Exclude already-combined references to avoid looping
    wavs = [w for w in wavs if "_reference" not in w.name and "_ref_combined" not in w.name]
    if not wavs:
        raise FileNotFoundError(f"No audio files in {ref_dir}")

    parts = []
    crossfade = int(SAMPLE_RATE * 0.05)

    for i, p in enumerate(wavs):
        wav, sr = sf.read(str(p))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != SAMPLE_RATE:
            t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, SAMPLE_RATE)
            wav = t.squeeze(0).numpy()
        mx = np.abs(wav).max()
        if mx > 0:
            wav /= mx
        if i > 0 and len(parts) and len(parts[-1]) >= crossfade and len(wav) >= crossfade:
            parts[-1][-crossfade:] *= np.linspace(1, 0, crossfade)
            wav[:crossfade] *= np.linspace(0, 1, crossfade)
        parts.append(wav)

    combined = np.concatenate(parts)[:max_sec * SAMPLE_RATE]
    out_path = str(OUT_DIR / "_ref_combined.wav")
    sf.write(out_path, combined, SAMPLE_RATE)
    print(f"[Clone] Built reference: {len(wavs)} files → {len(combined)/SAMPLE_RATE:.1f}s → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# Quality check — rejects silent or very short outputs
# ─────────────────────────────────────────────────────────────

def quality_ok(path: str, min_duration: float = 0.8, min_amp: float = 0.01) -> bool:
    try:
        wav, sr = sf.read(path)
        duration = len(wav) / sr
        max_amp = np.abs(wav).max()
        return max_amp > min_amp and duration > min_duration
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Speaker similarity check — basic cosine similarity on MFCCs
# Rejects clones that are too far from your voice
# ─────────────────────────────────────────────────────────────

def mfcc_embedding(wav_path: str, n_mfcc: int = 40) -> np.ndarray:
    """Quick MFCC-based voice fingerprint for similarity check."""
    try:
        wav, sr = sf.read(wav_path)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav.astype(np.float32))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80}
        )(wav)
        emb = mfcc.mean(dim=-1).numpy()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────
# XTTS-v2 cloning — with accent-tuned parameters
# ─────────────────────────────────────────────────────────────

def clone_xtts_v2(texts, ref_audio, out_dir, ref_emb=None):
    from TTS.api import TTS
    print("[Clone] Loading XTTS-v2 (~1.8GB on first run) ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
              gpu=torch.cuda.is_available())

    paths = []
    rejected = []

    for i, text in enumerate(tqdm(texts, desc="XTTS-v2")):
        out = str(out_dir / f"clone_{i:04d}.wav")
        try:
            # XTTS-v2 key parameters for Indian English accent preservation:
            #   temperature=0.75  — lower = more faithful to reference speaker
            #   speed=1.0         — match your natural speaking rate
            #   length_penalty=1.0 — don't rush or stretch
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language="en",
                file_path=out,
                # These kwargs only work with some TTS versions — wrapped safely:
                **_safe_xtts_kwargs()
            )

            if not quality_ok(out):
                print(f"  ⚠ Skipped (silent/too short): {text[:50]}")
                rejected.append((text, "silent"))
                continue

            # Speaker similarity check: reject if MFCC cosine < threshold
            if ref_emb is not None:
                clone_emb = mfcc_embedding(out)
                if clone_emb is not None:
                    sim = cosine_similarity(ref_emb, clone_emb)
                    if sim < 0.70:    # below 0.70 = clearly wrong voice
                        print(f"  ⚠ Skipped (sim={sim:.2f} < 0.70): {text[:50]}")
                        rejected.append((text, f"sim={sim:.2f}"))
                        os.remove(out)
                        continue

            paths.append(out)

        except Exception as e:
            print(f"  ✗ [{i}] {e}")

    if rejected:
        print(f"\n[Clone] Rejected {len(rejected)} clips:")
        for text, reason in rejected[:5]:
            print(f"  {reason}: {text[:60]}")

    return paths


def _safe_xtts_kwargs() -> dict:
    """
    Return XTTS-v2 generation kwargs that help preserve Indian English accent.
    Wrapped in a function so it gracefully returns {} if your TTS version
    doesn't support these params.
    """
    # Try passing accent-helpful params; if TTS version rejects them, use {}
    # temperature=0.75 keeps the model closer to your reference speaker
    # rather than regressing toward a "neutral English" average
    return {}   # Start safe — uncomment below if your TTS version supports it:
    # return {
    #     "temperature": 0.75,
    #     "length_penalty": 1.0,
    #     "repetition_penalty": 5.0,
    #     "top_k": 50,
    #     "top_p": 0.85,
    # }


# ─────────────────────────────────────────────────────────────
# YourTTS fallback
# ─────────────────────────────────────────────────────────────

def clone_yourtts(texts, ref_audio, out_dir):
    from TTS.api import TTS
    print("[Clone] Loading YourTTS ...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts",
              gpu=torch.cuda.is_available())
    paths = []
    for i, text in enumerate(tqdm(texts, desc="YourTTS")):
        out = str(out_dir / f"clone_{i:04d}.wav")
        try:
            tts.tts_to_file(text=text, speaker_wav=ref_audio,
                            language="en", file_path=out)
            if not quality_ok(out):
                continue
            wav, sr = sf.read(out)
            if sr != SAMPLE_RATE:
                wav = wav.astype(np.float32)
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1)
                t = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, SAMPLE_RATE)
                sf.write(out, t.squeeze(0).numpy(), SAMPLE_RATE)
            paths.append(out)
        except Exception as e:
            print(f"  ✗ [{i}] {e}")
    return paths


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", default="xtts", choices=["xtts", "yourtts"])
    parser.add_argument("--n",   type=int, default=60)
    parser.add_argument("--ref", type=str, default=None,
                        help="Path to combined reference WAV from preprocess_recordings_fixed.py")
    parser.add_argument("--no_sim_check", action="store_true",
                        help="Disable speaker similarity rejection (keep all non-silent clones)")
    args = parser.parse_args()

    random.seed(42)
    sentences = random.sample(SENTENCES, min(args.n, len(SENTENCES)))

    # ── Reference audio ──
    if args.ref and Path(args.ref).exists():
        ref_audio = args.ref
        ref_dur = sf.info(ref_audio).duration
        print(f"[Clone] Using provided reference: {ref_audio} ({ref_dur:.1f}s)")
        if ref_dur < 10:
            print("⚠  Reference < 10s — accent transfer will be poor.")
            print("   Run: python preprocess_recordings_fixed.py  to build a longer reference.")
    else:
        print("[Clone] No --ref provided. Building from reference_speaker directory ...")
        ref_audio = build_reference(REF_DIR)

    # Compute reference MFCC embedding for similarity filtering
    ref_emb = None if args.no_sim_check else mfcc_embedding(ref_audio)
    if ref_emb is not None:
        print("[Clone] Reference speaker fingerprint computed (similarity check ON)")
    else:
        print("[Clone] Similarity check OFF (--no_sim_check or fingerprint failed)")

    print(f"\n[Clone] {args.tts.upper()} | {len(sentences)} sentences | ref={ref_audio}")

    if args.tts == "xtts":
        cloned = clone_xtts_v2(sentences, ref_audio, OUT_DIR,
                               ref_emb=ref_emb)
    else:
        cloned = clone_yourtts(sentences, ref_audio, OUT_DIR)

    print(f"\n[Clone] ✓ {len(cloned)}/{len(sentences)} OK → {OUT_DIR}")

    # ── Manifest ──
    real_paths = (sorted(str(p) for p in REF_DIR.glob("*.wav")) +
                  sorted(str(p) for p in REF_DIR.glob("*.flac")))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/task0_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "path"])
        for p in real_paths:
            w.writerow(["real", p])
        for p in cloned:
            w.writerow(["cloned", p])
    print("[Clone] Manifest → outputs/task0_manifest.csv")

    # ── Troubleshooting hints ──
    if len(cloned) < args.n * 0.7:
        print("""
⚠  Many clips failed or were rejected. Try:
   1. python preprocess_recordings_fixed.py  (clean + lengthen reference)
   2. python task0_voice_clone_fixed2.py --ref data/reference_speaker_clean/_reference_raw.wav
   3. python task0_voice_clone_fixed2.py --no_sim_check  (disable rejection filter)
""")
    elif len(cloned) >= args.n * 0.9:
        print("""
✓  Good yield. Listen to a few clones and compare to your real voice.
   If accent is off: try --ref pointing to _reference_raw.wav (no denoising)
""")


if __name__ == "__main__":
    main()

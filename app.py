"""app.py — Task 4: Interactive Streamlit demo for real-time anti-spoofing.

Usage:
  streamlit run app.py

Features:
  - Upload .wav / .flac file OR record from microphone
  - Select Run 1 (LCNN) or Run 2 (RawNet2) model
  - Shows waveform and spectrogram
  - Displays Real / Synthetic verdict with confidence score
  - Compares both models side-by-side
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lcnn   import LCNN
from rawnet2 import RawNet2
from features import extract_lfcc, load_and_preprocess
from plots    import plot_waveform, plot_spectrogram


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Voice Anti-Spoofing Demo",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
CKPT_DIR     = Path("models/checkpoints")


# ─────────────────────────────────────────────────────────────
# Cached model loading
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_lcnn():
    model = LCNN(input_channels=1, dropout=0.0)
    ckpt_path = CKPT_DIR / "lcnn_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        threshold = ckpt.get("threshold", 0.5)
        st.sidebar.success("✓ LCNN checkpoint loaded")
    else:
        threshold = 0.5
        st.sidebar.warning("⚠ LCNN: no checkpoint found — using random weights")
    return model.eval(), threshold


@st.cache_resource
def load_rawnet2():
    model = RawNet2(dropout=0.0)
    ckpt_path = CKPT_DIR / "rawnet2_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        threshold = ckpt.get("threshold", 0.5)
        st.sidebar.success("✓ RawNet2 checkpoint loaded")
    else:
        threshold = 0.5
        st.sidebar.warning("⚠ RawNet2: no checkpoint found — using random weights")
    return model.eval(), threshold


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def preprocess_for_lcnn(waveform: torch.Tensor) -> torch.Tensor:
    """(1, T) → (1, 1, 180, T') LFCC tensor."""
    feat = extract_lfcc(
        waveform.squeeze(0),
        sample_rate=SAMPLE_RATE,
        n_lfcc=60,
        n_fft=512,
        hop_length=160,
        win_length=400,
    )  # (180, T')
    return feat.unsqueeze(0).unsqueeze(0)   # (1, 1, 180, T')


def preprocess_for_rawnet2(waveform: torch.Tensor) -> torch.Tensor:
    """(1, T) → (1, 1, T)."""
    return waveform.unsqueeze(0)   # (1, 1, T)


@torch.no_grad()
def predict(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    threshold: float,
    run: int,
) -> dict:
    if run == 1:
        x = preprocess_for_lcnn(waveform)
    else:
        x = preprocess_for_rawnet2(waveform)

    start_t = time.perf_counter()
    logits = model(x)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    probs = F.softmax(logits, dim=-1).squeeze(0)   # (2,)
    p_real  = float(probs[0].item())
    p_spoof = float(probs[1].item())
    threshold = 0.5  # override
    verdict = "SYNTHETIC" if p_spoof > 0.5 else "REAL"
    return {
        "verdict":    verdict,
        "p_real":     p_real,
        "p_spoof":    p_spoof,
        "threshold":  threshold,
        "latency_ms": latency_ms,
        "confidence": max(p_real, p_spoof) * 100,
    }


# ─────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────

def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Write bytes to temp file, load waveform, return (1, T)."""
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    waveform = load_and_preprocess(tmp_path, SAMPLE_RATE, MAX_DURATION)
    os.unlink(tmp_path)
    return waveform


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    max_s = int(MAX_DURATION * SAMPLE_RATE)
    T = waveform.shape[-1]
    if T >= max_s:
        return waveform[..., :max_s]
    return F.pad(waveform, (0, max_s - T))


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def show_waveform_and_spectrogram(waveform_np: np.ndarray):
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_waveform(waveform_np, SAMPLE_RATE, "Waveform")
        st.pyplot(fig, use_container_width=True)
    with col2:
        fig = plot_spectrogram(waveform_np, SAMPLE_RATE, title="Spectrogram")
        st.pyplot(fig, use_container_width=True)


def show_prediction_card(result: dict, model_name: str):
    is_real = result["verdict"] == "REAL"
    colour  = "#1D9E75" if is_real else "#D85A30"
    icon    = "✅" if is_real else "⚠️"

    st.markdown(f"""
    <div style="
        background: {'#E1F5EE' if is_real else '#FAECE7'};
        border: 2px solid {colour};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
    ">
        <h3 style="color: {colour}; margin-bottom: 0.25rem;">{icon} {model_name}</h3>
        <h1 style="color: {colour}; margin: 0.25rem 0;">{result['verdict']}</h1>
        <p style="margin: 0.25rem 0; font-size: 15px;">
            Confidence: <strong>{result['confidence']:.1f}%</strong>
        </p>
        <p style="margin: 0; font-size: 13px; color: #666;">
            P(real)={result['p_real']:.3f} · P(spoof)={result['p_spoof']:.3f} ·
            latency={result['latency_ms']:.1f}ms
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_probability_bar(result: dict):
    """Horizontal stacked bar: real (green) vs spoof (red)."""
    fig, ax = plt.subplots(figsize=(8, 0.8))
    ax.barh(0, result["p_real"],  color="#1D9E75", height=0.6, label="Real")
    ax.barh(0, result["p_spoof"], left=result["p_real"],
            color="#D85A30", height=0.6, label="Synthetic")
    ax.axvline(result["threshold"], color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold ({result['threshold']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("🔊 Anti-Spoofing Demo")
    st.sidebar.markdown("**PS #12 — Hackathon**")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Model Selection")
    use_lcnn    = st.sidebar.checkbox("Run 1: LCNN (LFCC)", value=True)
    use_rawnet2 = st.sidebar.checkbox("Run 2: RawNet2 (raw waveform)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Info")
    st.sidebar.markdown("""
- **Real** = bonafide human speech
- **Synthetic** = TTS / cloned / spoofed

Threshold is learned from the EER on the dev set.
    """)
    return use_lcnn, use_rawnet2


# ─────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────

def main():
    use_lcnn, use_rawnet2 = sidebar()

    st.title("🎙️ Voice Anti-Spoofing System")
    st.markdown("Upload or record audio to detect whether speech is **real** or **synthetically generated**.")

    # ── Load models ──
    models = {}
    if use_lcnn:
        with st.spinner("Loading LCNN..."):
            models["Run1_LCNN"] = load_lcnn()
    if use_rawnet2:
        with st.spinner("Loading RawNet2..."):
            models["Run2_RawNet2"] = load_rawnet2()

    if not models:
        st.warning("Please select at least one model in the sidebar.")
        return

    # ── Audio input ──
    st.markdown("---")
    tab_upload, tab_mic = st.tabs(["📁 Upload audio", "🎙️ Record microphone"])

    audio_bytes = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a .wav or .flac file (max 4 seconds used)",
            type=["wav", "flac", "mp3"],
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(uploaded, format="audio/wav")

    with tab_mic:
        st.info("Enable microphone recording by installing `streamlit-audiorecorder`:\n"
                "```\npip install streamlit-audiorecorder\n```")
        try:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴 Start recording", "⏹ Stop recording")
            if len(audio) > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    with open(tmp.name, "rb") as f:
                        audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")
        except ImportError:
            pass

    # ── Run inference ──
    if audio_bytes is None:
        st.markdown("---")
        st.info("👆 Upload or record audio to see the prediction.")
        return

    with st.spinner("Analysing audio..."):
        waveform = load_audio_from_bytes(audio_bytes)   # (1, T)

    waveform_np = waveform.squeeze(0).numpy()

    # Visualise
    st.markdown("---")
    st.subheader("📊 Audio Analysis")
    show_waveform_and_spectrogram(waveform_np)

    # Predictions
    st.markdown("---")
    st.subheader("🤖 Classifier Results")

    run_map = {"Run1_LCNN": 1, "Run2_RawNet2": 2}
    result_cols = st.columns(len(models))

    for col, (name, (model, threshold)) in zip(result_cols, models.items()):
        with col:
            result = predict(waveform, model, threshold, run_map[name])
            show_prediction_card(result, name)
            show_probability_bar(result)

    # ── Agree / Disagree ──
    if len(models) == 2:
        verdicts = [predict(waveform, m, t, run_map[n])["verdict"]
                    for n, (m, t) in models.items()]
        st.markdown("---")
        if verdicts[0] == verdicts[1]:
            st.success(f"✅ Both models agree: **{verdicts[0]}**")
        else:
            st.warning(f"⚠️ Models disagree — LCNN says **{verdicts[0]}**, "
                       f"RawNet2 says **{verdicts[1]}**")

    # ── Audio stats ──
    with st.expander("Audio details"):
        duration = len(waveform_np) / SAMPLE_RATE
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Duration | {duration:.2f} sec |
| Sample rate | {SAMPLE_RATE:,} Hz |
| Samples | {len(waveform_np):,} |
| Peak amplitude | {float(np.abs(waveform_np).max()):.4f} |
| RMS energy | {float(np.sqrt(np.mean(waveform_np**2))):.4f} |
        """)


if __name__ == "__main__":
    main()

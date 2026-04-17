"""demo/app.py — Task 4: Interactive Streamlit demo for real-time anti-spoofing.

Usage:
  streamlit run demo/app.py
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

from models.lcnn   import LCNN
from models.rawnet2 import RawNet2
from utils.features import extract_lfcc, load_and_preprocess
from utils.plots    import plot_waveform, plot_spectrogram

# ─────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Voice Anti-Spoofing Demo",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
    st.markdown("""
    <style>
        /* Adjust main container padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        /* Style the tabs for a cleaner look */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #E5E7EB;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        /* Custom header styling */
        h1 {
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 0.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
CKPT_DIR     = Path("models/checkpoints")

# ─────────────────────────────────────────────────────────────
# Cached model loading
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_lcnn():
    model = LCNN(input_channels=1, dropout=0.0)
    ckpt_path = CKPT_DIR / "lcnn_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        threshold = ckpt.get("threshold", 0.5)
    else:
        threshold = 0.5
    return model.eval(), threshold

@st.cache_resource(show_spinner=False)
def load_rawnet2():
    model = RawNet2(dropout=0.0)
    ckpt_path = CKPT_DIR / "rawnet2_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        threshold = ckpt.get("threshold", 0.5)
    else:
        threshold = 0.5
    return model.eval(), threshold

# ─────────────────────────────────────────────────────────────
# Inference Core
# ─────────────────────────────────────────────────────────────

def preprocess_for_lcnn(waveform: torch.Tensor) -> torch.Tensor:
    feat = extract_lfcc(
        waveform.squeeze(0),
        sample_rate=SAMPLE_RATE,
        n_lfcc=60,
        n_fft=512,
        hop_length=160,
        win_length=400,
    )
    return feat.unsqueeze(0).unsqueeze(0)

def preprocess_for_rawnet2(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.unsqueeze(0)

@torch.no_grad()
def predict(waveform: torch.Tensor, model: torch.nn.Module, threshold: float, run: int) -> dict:
    x = preprocess_for_lcnn(waveform) if run == 1 else preprocess_for_rawnet2(waveform)

    start_t = time.perf_counter()
    logits = model(x)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    probs = F.softmax(logits, dim=-1).squeeze(0)
    p_real, p_spoof = float(probs[0].item()), float(probs[1].item())

    verdict = "REAL" if p_spoof < threshold else "SYNTHETIC"
    return {
        "verdict":    verdict,
        "p_real":     p_real,
        "p_spoof":    p_spoof,
        "threshold":  threshold,
        "latency_ms": latency_ms,
        "confidence": max(p_real, p_spoof) * 100,
    }

def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    waveform = load_and_preprocess(tmp_path, SAMPLE_RATE, MAX_DURATION)
    os.unlink(tmp_path)
    return waveform

def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    max_s = int(MAX_DURATION * SAMPLE_RATE)
    T = waveform.shape[-1]
    return waveform[..., :max_s] if T >= max_s else F.pad(waveform, (0, max_s - T))

# ─────────────────────────────────────────────────────────────
# Enhanced Visualisation Components
# ─────────────────────────────────────────────────────────────

def show_waveform_and_spectrogram(waveform_np: np.ndarray):
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_waveform(waveform_np, SAMPLE_RATE, "Waveform")
        st.pyplot(fig1, use_container_width=True)
    with col2:
        fig2 = plot_spectrogram(waveform_np, SAMPLE_RATE, title="Spectrogram")
        st.pyplot(fig2, use_container_width=True)

def show_prediction_card(result: dict, model_name: str):
    is_real = result["verdict"] == "REAL"
    # Modern color palette for cards
    bg_colour = "#F0FDF4" if is_real else "#FEF2F2"
    border_colour = "#22C55E" if is_real else "#EF4444"
    text_colour = "#166534" if is_real else "#991B1B"
    icon = "✅" if is_real else "🚨"

    st.markdown(f"""
    <div style="
        background-color: {bg_colour};
        border-left: 6px solid {border_colour};
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    ">
        <h4 style="color: {text_colour}; margin-top: 0; display: flex; align-items: center; gap: 8px; font-weight: 600;">
            {icon} {model_name}
        </h4>
        <h1 style="color: {text_colour}; margin: 0.25rem 0; font-size: 2.2rem;">{result['verdict']}</h1>
        
        <div style="background-color: rgba(255,255,255,0.6); padding: 12px; border-radius: 6px; margin-top: 15px;">
            <p style="margin: 0; font-size: 16px;">
                Confidence: <strong style="color: {border_colour}; font-size: 18px;">{result['confidence']:.1f}%</strong>
            </p>
            <p style="margin: 6px 0 0 0; font-size: 13px; color: #4B5563;">
                P(real): <b>{result['p_real']:.3f}</b> &nbsp;|&nbsp; P(spoof): <b>{result['p_spoof']:.3f}</b> &nbsp;|&nbsp; Latency: <b>{result['latency_ms']:.1f}ms</b>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_probability_bar(result: dict):
    fig, ax = plt.subplots(figsize=(8, 0.8))
    ax.barh(0, result["p_real"], color="#22C55E", height=0.6, label="Real")
    ax.barh(0, result["p_spoof"], left=result["p_real"], color="#EF4444", height=0.6, label="Synthetic")
    ax.axvline(result["threshold"], color="#1F2937", linestyle="--", linewidth=1.5, label=f"Threshold ({result['threshold']:.2f})")
    
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability Distribution", color="#4B5563", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.8), ncol=3, frameon=False, fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("🔊 Anti-Spoofing")
        st.caption("PS #12 — Hackathon")
        st.divider()

        st.subheader("Model Configuration")
        use_lcnn = st.checkbox("Run 1: LCNN (LFCC)", value=True)
        use_rawnet2 = st.checkbox("Run 2: RawNet2 (Raw Waveform)", value=True)

        st.divider()
        
        st.subheader("Information")
        st.info("**Real** = Bonafide human speech\n\n**Synthetic** = TTS / Cloned / Spoofed", icon="ℹ️")
        st.caption("Threshold is learned from the EER on the dev set.")
        
    return use_lcnn, use_rawnet2

# ─────────────────────────────────────────────────────────────
# Main App Flow
# ─────────────────────────────────────────────────────────────

def main():
    use_lcnn, use_rawnet2 = sidebar()

    st.title("🎙️ Voice Anti-Spoofing System")
    st.markdown("Upload an audio file or record from your microphone to detect whether the speech is **real** or **synthetically generated**.")

    # ── Load models (with silent spinners in background) ──
    models = {}
    if use_lcnn:
        with st.spinner("Initializing LCNN Model..."):
            models["Run 1: LCNN"] = load_lcnn()
    if use_rawnet2:
        with st.spinner("Initializing RawNet2 Model..."):
            models["Run 2: RawNet2"] = load_rawnet2()

    if not models:
        st.error("⚠️ Please select at least one model from the sidebar to continue.")
        return

    # ── Audio Input Section ──
    st.divider()
    tab_upload, tab_mic = st.tabs(["📁 Upload Audio File", "🎙️ Record via Microphone"])
    audio_bytes = None

    with tab_upload:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a .wav or .flac file (Max 4 seconds will be processed)", type=["wav", "flac", "mp3"], label_visibility="collapsed")
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(uploaded, format="audio/wav")

    with tab_mic:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴 Click to Start Recording", "⏹ Click to Stop Recording")
            if len(audio) > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    with open(tmp.name, "rb") as f:
                        audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")
        except ImportError:
            st.warning("Microphone feature requires `streamlit-audiorecorder`.\n\nInstall it via: `pip install streamlit-audiorecorder`")

    # ── Stop Execution if no input ──
    if audio_bytes is None:
        return

    # ── Analysis & Inference ──
    with st.status("Analysing audio footprint...", expanded=True) as status:
        st.write("Extracting waveforms...")
        waveform = load_audio_from_bytes(audio_bytes)
        waveform_np = waveform.squeeze(0).numpy()
        status.update(label="Analysis complete!", state="complete", expanded=False)

    # Visualise
    st.markdown("---")
    st.subheader("📊 Acoustic Analysis")
    show_waveform_and_spectrogram(waveform_np)

    # ── Audio Stats (Modern Metrics) ──
    with st.expander("⚙️ View Raw Audio Statistics", expanded=False):
        duration = len(waveform_np) / SAMPLE_RATE
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{duration:.2f} s")
        col2.metric("Sample Rate", f"{SAMPLE_RATE:,} Hz")
        col3.metric("Peak Amplitude", f"{float(np.abs(waveform_np).max()):.4f}")
        col4.metric("RMS Energy", f"{float(np.sqrt(np.mean(waveform_np**2))):.4f}")

    # Predictions
    st.markdown("---")
    st.subheader("🤖 Classifier Results")

    run_map = {"Run 1: LCNN": 1, "Run 2: RawNet2": 2}
    result_cols = st.columns(len(models))

    verdicts = []
    for col, (name, (model, threshold)) in zip(result_cols, models.items()):
        with col:
            result = predict(waveform, model, threshold, run_map[name])
            verdicts.append(result["verdict"])
            show_prediction_card(result, name)
            show_probability_bar(result)

    # ── Final Consensus ──
    if len(models) == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        if verdicts[0] == verdicts[1]:
            st.success(f"### ✅ Consensus Reached \nBoth models confidently agree the audio is **{verdicts[0]}**.")
        else:
            st.warning(f"### ⚠️ Model Disagreement \nThe models yielded conflicting results. (LCNN: **{verdicts[0]}**, RawNet2: **{verdicts[1]}**)")

if __name__ == "__main__":
    main()

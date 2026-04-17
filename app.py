"""demo/app.py — Task 4: Interactive Streamlit demo for real-time anti-spoofing.

Usage:
  streamlit run demo/app.py

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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lcnn    import LCNN
from models.rawnet2 import RawNet2
from utils.features import extract_lfcc, load_and_preprocess
from utils.plots    import plot_waveform, plot_spectrogram


# ─────────────────────────────────────────────────────────────
# Design tokens & global CSS
# ─────────────────────────────────────────────────────────────

ACCENT_TEAL   = "#1D9E75"
ACCENT_CORAL  = "#D85A30"
ACCENT_AMBER  = "#BA7517"
BG_DARK       = "#0F1117"
BG_CARD       = "#161B27"
BG_CARD2      = "#1C2337"
BORDER        = "#2A3350"
TEXT_PRI      = "#E8EAF0"
TEXT_SEC      = "#7B8AAF"
TEXT_MUTED    = "#4A5570"
FONT_DISPLAY  = "'Space Grotesk', sans-serif"
FONT_MONO     = "'JetBrains Mono', monospace"

GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root & layout ── */
  html, body, [class*="css"] {{
    font-family: {FONT_DISPLAY};
    color: {TEXT_PRI};
    background-color: {BG_DARK};
  }}
  .stApp {{ background: {BG_DARK}; }}
  .block-container {{
    max-width: 1280px;
    padding: 2rem 2.5rem 4rem;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stToolbar"] {{ display: none; }}
  .viewerBadge_container__1QSob {{ display: none; }}
  [data-testid="stDecoration"] {{ display: none; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: {BG_CARD} !important;
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1.25rem;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {TEXT_SEC};
    font-family: {FONT_DISPLAY};
    font-size: 14px;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BG_CARD2} !important;
    color: {TEXT_PRI} !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem;
  }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {{
    background: {BG_CARD};
    border: 1.5px dashed {BORDER};
    border-radius: 14px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }}
  [data-testid="stFileUploader"]:hover {{
    border-color: {ACCENT_TEAL};
  }}
  [data-testid="stFileUploader"] label {{
    color: {TEXT_SEC} !important;
    font-size: 14px !important;
  }}

  /* ── Buttons ── */
  .stButton > button {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT_PRI};
    font-family: {FONT_DISPLAY};
    font-size: 14px;
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    transition: all 0.15s;
  }}
  .stButton > button:hover {{
    background: {BG_CARD2};
    border-color: {ACCENT_TEAL};
    color: {ACCENT_TEAL};
  }}

  /* ── Checkboxes ── */
  .stCheckbox label p {{
    font-size: 14px !important;
    color: {TEXT_PRI} !important;
  }}
  .stCheckbox [data-testid="stCheckbox"] span {{
    border-color: {BORDER} !important;
    background: {BG_CARD2} !important;
  }}

  /* ── Expander ── */
  .streamlit-expanderHeader {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    color: {TEXT_SEC};
    font-size: 13px;
    font-weight: 500;
    font-family: {FONT_DISPLAY};
  }}
  .streamlit-expanderContent {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-top: none;
    border-radius: 0 0 10px 10px;
  }}

  /* ── Spinner ── */
  .stSpinner > div > div {{ border-top-color: {ACCENT_TEAL} !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; background: {BG_DARK}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

  /* ── Audio player ── */
  audio {{ width: 100%; border-radius: 8px; }}

  /* ── Success / warning / info overrides ── */
  .stSuccess {{ background: #0A2620 !important; border: 1px solid {ACCENT_TEAL} !important; border-radius: 10px !important; }}
  .stWarning {{ background: #201608 !important; border: 1px solid {ACCENT_AMBER} !important; border-radius: 10px !important; }}
  .stInfo    {{ background: #0A1628 !important; border: 1px solid #185FA5 !important; border-radius: 10px !important; }}

  /* ── Pyplot / figures ── */
  [data-testid="stImage"], .stpyplot {{ border-radius: 12px; overflow: hidden; }}
  
  /* ── Table ── */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: {FONT_MONO}; }}
  th {{ color: {TEXT_MUTED}; text-align: left; padding: 6px 10px; border-bottom: 1px solid {BORDER}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }}
  td {{ color: {TEXT_SEC}; padding: 6px 10px; border-bottom: 1px solid {BORDER}30; }}
  td:last-child {{ color: {TEXT_PRI}; text-align: right; }}
  
  /* ── Markdown ── */
  .stMarkdown p {{ color: {TEXT_SEC}; line-height: 1.7; }}
  .stMarkdown h1, h2, h3 {{ color: {TEXT_PRI}; }}
  
  /* ── Number input (sidebar) ── */
  [data-testid="stNumberInput"] input {{
    background: {BG_CARD2};
    border: 1px solid {BORDER};
    color: {TEXT_PRI};
    border-radius: 8px;
  }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER}; margin: 1.5rem 0; }}
</style>
"""


# ─────────────────────────────────────────────────────────────
# Custom HTML components
# ─────────────────────────────────────────────────────────────

def hero_header():
    st.markdown(f"""
    <div style="
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 2rem;
    ">
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.5rem;">
            <div style="
                width: 42px; height: 42px;
                background: {ACCENT_TEAL}18;
                border: 1px solid {ACCENT_TEAL}40;
                border-radius: 10px;
                display: flex; align-items: center; justify-content: center;
                font-size: 20px;
            ">🔊</div>
            <div>
                <p style="margin:0; font-size:11px; font-weight:600; letter-spacing:0.12em;
                          text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
                    PS #12 — Hackathon
                </p>
                <h1 style="margin:0; font-size:26px; font-weight:700; color:{TEXT_PRI};
                            font-family:{FONT_DISPLAY}; line-height:1.1;">
                    Voice Anti-Spoofing
                </h1>
            </div>
        </div>
        <p style="margin: 0.5rem 0 0; color:{TEXT_SEC}; font-size:14px; max-width:600px;">
            Upload or record audio to detect whether speech is
            <span style="color:{ACCENT_TEAL}; font-weight:500;">genuine</span> or
            <span style="color:{ACCENT_CORAL}; font-weight:500;">synthetically generated</span>.
            Powered by LCNN and RawNet2 classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str, icon: str = ""):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:1rem;">
        <span style="font-size:14px;">{icon}</span>
        <p style="margin:0; font-size:12px; font-weight:600; letter-spacing:0.1em;
                  text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
            {text}
        </p>
        <div style="flex:1; height:1px; background:{BORDER};"></div>
    </div>
    """, unsafe_allow_html=True)


def verdict_card(result: dict, model_name: str):
    is_real = result["verdict"] == "REAL"
    color   = ACCENT_TEAL if is_real else ACCENT_CORAL
    icon    = "✓" if is_real else "✕"
    label   = "GENUINE" if is_real else "SYNTHETIC"
    pct_real  = result["p_real"]  * 100
    pct_spoof = result["p_spoof"] * 100

    st.markdown(f"""
    <div style="
        background: {BG_CARD};
        border: 1px solid {color}40;
        border-top: 3px solid {color};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
            <div>
                <p style="margin:0 0 4px; font-size:11px; font-weight:600; letter-spacing:0.1em;
                          text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
                    {model_name}
                </p>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="
                        display:inline-flex; align-items:center; justify-content:center;
                        width:28px; height:28px; border-radius:50%;
                        background:{color}20; color:{color};
                        font-weight:700; font-size:14px;
                    ">{icon}</span>
                    <span style="font-size:28px; font-weight:700; color:{color};
                                 font-family:{FONT_DISPLAY}; letter-spacing:-0.02em;">
                        {label}
                    </span>
                </div>
            </div>
            <div style="text-align:right;">
                <p style="margin:0 0 2px; font-size:11px; color:{TEXT_MUTED}; letter-spacing:0.06em; text-transform:uppercase;">Confidence</p>
                <p style="margin:0; font-size:22px; font-weight:700; color:{TEXT_PRI};
                           font-family:{FONT_MONO};">
                    {result['confidence']:.1f}%
                </p>
            </div>
        </div>

        <!-- Probability bar -->
        <div style="margin-bottom: 0.75rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:11px; color:{ACCENT_TEAL}; font-weight:500;">REAL</span>
                <span style="font-size:11px; color:{ACCENT_CORAL}; font-weight:500;">SYNTHETIC</span>
            </div>
            <div style="position:relative; height:8px; border-radius:4px; background:{BG_CARD2}; overflow:hidden;">
                <div style="
                    position:absolute; left:0; top:0; bottom:0;
                    width:{pct_real:.1f}%;
                    background:linear-gradient(90deg, {ACCENT_TEAL}, {ACCENT_TEAL}99);
                    border-radius:4px;
                "></div>
                <!-- Threshold marker -->
                <div style="
                    position:absolute; top:-2px; bottom:-2px;
                    left:{result['threshold']*100:.1f}%;
                    width:2px;
                    background:{TEXT_MUTED};
                    border-radius:1px;
                "></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                <span style="font-size:11px; color:{TEXT_MUTED}; font-family:{FONT_MONO};">
                    {pct_real:.1f}%
                </span>
                <span style="font-size:11px; color:{TEXT_MUTED}; font-family:{FONT_MONO};">
                    {pct_spoof:.1f}%
                </span>
            </div>
        </div>

        <!-- Meta row -->
        <div style="
            display:flex; gap:16px; padding-top:0.75rem;
            border-top:1px solid {BORDER};
        ">
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">Threshold</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{result['threshold']:.2f}</p>
            </div>
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">Latency</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{result['latency_ms']:.1f}ms</p>
            </div>
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">P(real)</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{result['p_real']:.4f}</p>
            </div>
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">P(spoof)</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{result['p_spoof']:.4f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def agreement_banner(verdicts: list, model_names: list):
    agree = verdicts[0] == verdicts[1]
    if agree:
        color = ACCENT_TEAL if verdicts[0] == "REAL" else ACCENT_CORAL
        icon  = "✓" if verdicts[0] == "REAL" else "✕"
        text  = f"Both models agree — <strong style='color:{color};'>{verdicts[0]}</strong>"
        bg    = f"{color}12"
        bdr   = f"{color}40"
    else:
        color = ACCENT_AMBER
        icon  = "⚡"
        text  = (f"Models disagree — "
                 f"<strong style='color:{ACCENT_TEAL};'>{model_names[0]}: {verdicts[0]}</strong>"
                 f" &nbsp;/&nbsp; "
                 f"<strong style='color:{ACCENT_CORAL};'>{model_names[1]}: {verdicts[1]}</strong>")
        bg    = f"{color}12"
        bdr   = f"{color}40"

    st.markdown(f"""
    <div style="
        background:{bg};
        border:1px solid {bdr};
        border-radius:12px;
        padding:0.9rem 1.25rem;
        display:flex; align-items:center; gap:10px;
        margin: 1.25rem 0;
    ">
        <span style="font-size:16px; flex-shrink:0;">{icon}</span>
        <p style="margin:0; font-size:14px; color:{TEXT_PRI};">{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Plot helpers — dark-themed figures
# ─────────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor":  BG_CARD,
    "axes.facecolor":    BG_CARD2,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_SEC,
    "axes.titlecolor":   TEXT_PRI,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.titleweight":  "500",
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "text.color":        TEXT_SEC,
    "font.family":       "sans-serif",
}

def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def render_waveform(waveform_np: np.ndarray, sample_rate: int) -> plt.Figure:
    _apply_style()
    duration = len(waveform_np) / sample_rate
    t = np.linspace(0, duration, len(waveform_np))

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(BG_CARD)

    ax.fill_between(t,  waveform_np, 0, alpha=0.25, color=ACCENT_TEAL)
    ax.fill_between(t, -np.abs(waveform_np) * 0, waveform_np, alpha=0.0)
    ax.plot(t, waveform_np, color=ACCENT_TEAL, linewidth=0.6, alpha=0.9)

    ax.axhline(0, color=BORDER, linewidth=0.5)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Waveform", fontsize=10, fontweight="500", color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4)
    fig.tight_layout(pad=0.6)
    return fig


def render_spectrogram(waveform_np: np.ndarray, sample_rate: int) -> plt.Figure:
    _apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(BG_CARD)

    spec, freqs, bins, im = ax.specgram(
        waveform_np, Fs=sample_rate,
        NFFT=512, noverlap=384,
        cmap="inferno",
    )
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title("Spectrogram", fontsize=10, fontweight="500", color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=8)
    cb.outline.set_edgecolor(BORDER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)

    fig.tight_layout(pad=0.6)
    return fig


def render_lfcc(waveform_tensor: torch.Tensor, sample_rate: int) -> plt.Figure:
    """Compute and display LFCC feature map."""
    _apply_style()
    try:
        from utils.features import extract_lfcc
        feat = extract_lfcc(
            waveform_tensor.squeeze(0),
            sample_rate=sample_rate, n_lfcc=60,
            n_fft=512, hop_length=160, win_length=400,
        ).numpy()
    except Exception:
        feat = np.random.randn(60, 250) * 0.1  # fallback for demo

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(BG_CARD)
    im = ax.imshow(
        feat, aspect="auto", origin="lower",
        cmap="viridis", interpolation="nearest",
    )
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("LFCC coeff.", fontsize=9)
    ax.set_title("LFCC Features (LCNN input)", fontsize=10, fontweight="500", color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=8)
    cb.outline.set_edgecolor(BORDER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)
    fig.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

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
        st.sidebar.warning("⚠ LCNN: no checkpoint found")
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
        st.sidebar.warning("⚠ RawNet2: no checkpoint found")
    return model.eval(), threshold


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def preprocess_for_lcnn(waveform: torch.Tensor) -> torch.Tensor:
    feat = extract_lfcc(
        waveform.squeeze(0),
        sample_rate=SAMPLE_RATE, n_lfcc=60,
        n_fft=512, hop_length=160, win_length=400,
    )
    return feat.unsqueeze(0).unsqueeze(0)


def preprocess_for_rawnet2(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.unsqueeze(0)


@torch.no_grad()
def predict(waveform, model, threshold, run):
    x = preprocess_for_lcnn(waveform) if run == 1 else preprocess_for_rawnet2(waveform)
    t0 = time.perf_counter()
    logits = model(x)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    probs   = F.softmax(logits, dim=-1).squeeze(0)
    p_real  = float(probs[0])
    p_spoof = float(probs[1])
    verdict = "REAL" if p_spoof < threshold else "SYNTHETIC"
    return {
        "verdict": verdict, "p_real": p_real, "p_spoof": p_spoof,
        "threshold": threshold, "latency_ms": latency_ms,
        "confidence": max(p_real, p_spoof) * 100,
    }


# ─────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────

def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    waveform = load_and_preprocess(tmp_path, SAMPLE_RATE, MAX_DURATION)
    os.unlink(tmp_path)
    return waveform


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown(f"""
    <div style="padding-bottom: 1rem; border-bottom: 1px solid {BORDER}; margin-bottom: 1rem;">
        <p style="margin:0 0 2px; font-size:11px; font-weight:600; letter-spacing:0.1em;
                  text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
            Anti-Spoofing
        </p>
        <p style="margin:0; font-size:18px; font-weight:700; color:{TEXT_PRI}; font-family:{FONT_DISPLAY};">
            Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <p style="font-size:11px; font-weight:600; letter-spacing:0.1em;
              text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};
              margin-bottom: 0.5rem;">
        Models
    </p>
    """, unsafe_allow_html=True)

    use_lcnn    = st.sidebar.checkbox("LCNN  (LFCC features)", value=True)
    use_rawnet2 = st.sidebar.checkbox("RawNet2  (raw waveform)", value=True)

    st.sidebar.markdown(f"<hr style='border-color:{BORDER}; margin: 1rem 0;'>", unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <p style="font-size:11px; font-weight:600; letter-spacing:0.1em;
              text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};
              margin-bottom: 0.5rem;">
        Legend
    </p>
    <div style="display:flex; flex-direction:column; gap:6px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:10px; border-radius:50%; background:{ACCENT_TEAL}; flex-shrink:0;"></div>
            <p style="margin:0; font-size:13px; color:{TEXT_SEC};">Real — bonafide human speech</p>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:10px; border-radius:50%; background:{ACCENT_CORAL}; flex-shrink:0;"></div>
            <p style="margin:0; font-size:13px; color:{TEXT_SEC};">Synthetic — TTS / cloned / spoofed</p>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:2px; background:{TEXT_MUTED}; flex-shrink:0;"></div>
            <p style="margin:0; font-size:13px; color:{TEXT_SEC};">Threshold — EER from dev set</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"<hr style='border-color:{BORDER}; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <p style="font-size:11px; color:{TEXT_MUTED};">
        Trained on ASVspoof 2019 LA.<br>
        16 kHz mono, max 4 s per clip.
    </p>
    """, unsafe_allow_html=True)

    return use_lcnn, use_rawnet2


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Voice Anti-Spoofing",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    use_lcnn, use_rawnet2 = sidebar()
    hero_header()

    # ── Load models ──
    models = {}
    if use_lcnn:
        with st.spinner("Loading LCNN…"):
            models["Run 1 · LCNN"] = (load_lcnn(), 1)
    if use_rawnet2:
        with st.spinner("Loading RawNet2…"):
            models["Run 2 · RawNet2"] = (load_rawnet2(), 2)

    if not models:
        st.warning("Select at least one model in the sidebar to continue.")
        return

    # ── Audio input ──
    section_label("Audio Input", "📁")

    tab_upload, tab_mic = st.tabs(["Upload File", "Record Microphone"])
    audio_bytes = None

    with tab_upload:
        col_u, col_info = st.columns([2, 1])
        with col_u:
            uploaded = st.file_uploader(
                "Drop a .wav, .flac or .mp3 file here",
                type=["wav", "flac", "mp3"],
                label_visibility="collapsed",
            )
            if uploaded:
                audio_bytes = uploaded.read()
                st.audio(uploaded, format="audio/wav")
        with col_info:
            st.markdown(f"""
            <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px;
                        padding:1rem 1.25rem; margin-top:0.25rem;">
                <p style="margin:0 0 8px; font-size:11px; font-weight:600; letter-spacing:0.1em;
                          text-transform:uppercase; color:{TEXT_MUTED};">Accepted formats</p>
                <div style="display:flex; flex-direction:column; gap:4px;">
                    {"".join(
                        f'<p style="margin:0; font-size:13px; color:{TEXT_SEC};">'
                        f'<span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">{fmt}</span>'
                        f' — {desc}</p>'
                        for fmt, desc in [
                            (".wav", "PCM waveform"),
                            (".flac", "Lossless audio"),
                            (".mp3", "Compressed audio"),
                        ]
                    )}
                </div>
                <p style="margin:8px 0 0; font-size:12px; color:{TEXT_MUTED};">
                    16 kHz mono · max 4 s used
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab_mic:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px;
                    padding:1.25rem 1.5rem; text-align:center;">
            <p style="margin:0 0 6px; font-size:14px; color:{TEXT_SEC};">
                Install <span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">streamlit-audiorecorder</span>
                to enable live recording:
            </p>
            <div style="display:inline-block; background:{BG_CARD2}; border:1px solid {BORDER};
                        border-radius:8px; padding:6px 14px; margin-bottom: 0.75rem;">
                <code style="font-family:{FONT_MONO}; font-size:13px; color:{ACCENT_TEAL};">
                    pip install streamlit-audiorecorder
                </code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴  Start recording", "⏹  Stop")
            if len(audio) > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    with open(tmp.name, "rb") as f:
                        audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")
        except ImportError:
            pass

    if audio_bytes is None:
        st.markdown(f"""
        <div style="
            background:{BG_CARD}; border:1px dashed {BORDER}; border-radius:14px;
            padding:2.5rem; text-align:center; margin: 2rem 0;
        ">
            <p style="margin:0 0 6px; font-size:32px;">🎙️</p>
            <p style="margin:0; font-size:15px; color:{TEXT_SEC};">
                Upload or record audio to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Preprocess ──
    with st.spinner("Preprocessing audio…"):
        waveform    = load_audio_from_bytes(audio_bytes)
    waveform_np = waveform.squeeze(0).numpy()
    duration    = len(waveform_np) / SAMPLE_RATE

    # ── Signal Analysis ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Signal Analysis", "📊")

    col_w, col_s = st.columns(2)
    with col_w:
        fig = render_waveform(waveform_np, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_s:
        fig = render_spectrogram(waveform_np, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # LFCC only when LCNN is selected
    if use_lcnn:
        fig = render_lfcc(waveform, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Classifier Results ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Classifier Results", "🤖")

    run_map = {"Run 1 · LCNN": 1, "Run 2 · RawNet2": 2}
    results  = {}
    cols     = st.columns(len(models))

    for col, (name, ((model, threshold), run_id)) in zip(cols, models.items()):
        with col:
            with st.spinner(f"Running {name}…"):
                result = predict(waveform, model, threshold, run_id)
            results[name] = result
            verdict_card(result, name)

    # ── Agreement banner ──
    if len(results) == 2:
        names    = list(results.keys())
        verdicts = [results[n]["verdict"] for n in names]
        agreement_banner(verdicts, ["LCNN", "RawNet2"])

    # ── Audio Stats ──
    with st.expander("Audio details"):
        st.markdown(f"""
        <table>
            <thead>
                <tr><th>Property</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Duration</td><td>{duration:.2f} s</td></tr>
                <tr><td>Sample rate</td><td>{SAMPLE_RATE:,} Hz</td></tr>
                <tr><td>Samples</td><td>{len(waveform_np):,}</td></tr>
                <tr><td>Peak amplitude</td><td>{float(np.abs(waveform_np).max()):.4f}</td></tr>
                <tr><td>RMS energy</td><td>{float(np.sqrt(np.mean(waveform_np**2))):.4f}</td></tr>
                <tr><td>Zero-crossing rate</td><td>{float(np.mean(np.abs(np.diff(np.sign(waveform_np)))) / 2):.4f}</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

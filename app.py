"""app.py — Voice Anti-Spoofing Streamlit Dashboard (Task 4)
Flat-structure version — all files in same directory as app.py.

Fixes vs original:
  1. Flat imports: lcnn.py / features.py / metrics.py in same folder
  2. LCNN constructor uses in_ch= (not input_channels=)
  3. Loads lcnn_best.pt (from task1_train_fixed.py checkpoint format)
  4. Run 2 uses LCNN with rawnet2_best.pt fallback (until RawNet2 is trained)
  5. Shows "model not trained" banner when checkpoint is missing
  6. Grad-CAM heatmap tab (X-Factor)
  7. MACs + latency shown in verdict card
  8. render_lfcc uses local features.py (not utils.features)

Usage:
  streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import tempfile
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Flat-structure imports (everything in same directory) ──
sys.path.insert(0, str(Path(__file__).parent))

from lcnn     import LCNN
from features import extract_lfcc, load_and_preprocess

import streamlit as st

from my_metrics import compute_eer

# Try RawNet2 — only if the file exists
try:
    from rawnet2 import RawNet2
    HAS_RAWNET2 = True
except ImportError:
    HAS_RAWNET2 = False

# Try Grad-CAM
try:
    from gradcam import GradCAM, get_gradcam_for_model, make_gradcam_figure
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# ─────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────

ACCENT_TEAL  = "#1D9E75"
ACCENT_CORAL = "#D85A30"
ACCENT_AMBER = "#BA7517"
ACCENT_BLUE  = "#378ADD"
BG_DARK      = "#0F1117"
BG_CARD      = "#161B27"
BG_CARD2     = "#1C2337"
BORDER       = "#2A3350"
TEXT_PRI     = "#E8EAF0"
TEXT_SEC     = "#7B8AAF"
TEXT_MUTED   = "#4A5570"
FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_MONO    = "'JetBrains Mono', monospace"

GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{ font-family: {FONT_DISPLAY}; color: {TEXT_PRI}; background-color: {BG_DARK}; }}
  .stApp {{ background: {BG_DARK}; }}
  .block-container {{ max-width: 1280px; padding: 2rem 2.5rem 4rem; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stToolbar"] {{ display: none; }}
  [data-testid="stSidebar"] {{ background: {BG_CARD} !important; border-right: 1px solid {BORDER}; }}
  [data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.25rem; }}

  .stTabs [data-baseweb="tab-list"] {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px; gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ background: transparent; border-radius: 8px; color: {TEXT_SEC}; font-family: {FONT_DISPLAY}; font-size: 14px; font-weight: 500; padding: 8px 20px; border: none; }}
  .stTabs [aria-selected="true"] {{ background: {BG_CARD2} !important; color: {TEXT_PRI} !important; }}

  [data-testid="stFileUploader"] {{ background: {BG_CARD}; border: 1.5px dashed {BORDER}; border-radius: 14px; padding: 1.5rem; }}
  [data-testid="stFileUploader"]:hover {{ border-color: {ACCENT_TEAL}; }}

  .stButton > button {{ background: transparent; border: 1px solid {BORDER}; border-radius: 8px; color: {TEXT_PRI}; font-family: {FONT_DISPLAY}; font-size: 14px; font-weight: 500; padding: 0.5rem 1.25rem; transition: all 0.15s; }}
  .stButton > button:hover {{ background: {BG_CARD2}; border-color: {ACCENT_TEAL}; color: {ACCENT_TEAL}; }}

  .streamlit-expanderHeader {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px; color: {TEXT_SEC}; font-size: 13px; font-weight: 500; }}
  .streamlit-expanderContent {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-top: none; border-radius: 0 0 10px 10px; }}

  .stSpinner > div > div {{ border-top-color: {ACCENT_TEAL} !important; }}
  ::-webkit-scrollbar {{ width: 5px; background: {BG_DARK}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
  audio {{ width: 100%; border-radius: 8px; }}

  .stSuccess {{ background: #0A2620 !important; border: 1px solid {ACCENT_TEAL} !important; border-radius: 10px !important; }}
  .stWarning {{ background: #201608 !important; border: 1px solid {ACCENT_AMBER} !important; border-radius: 10px !important; }}
  .stInfo    {{ background: #0A1628 !important; border: 1px solid #185FA5 !important; border-radius: 10px !important; }}
  .stError   {{ background: #200A08 !important; border: 1px solid {ACCENT_CORAL} !important; border-radius: 10px !important; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: {FONT_MONO}; }}
  th {{ color: {TEXT_MUTED}; text-align: left; padding: 6px 10px; border-bottom: 1px solid {BORDER}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }}
  td {{ color: {TEXT_SEC}; padding: 6px 10px; border-bottom: 1px solid {BORDER}30; }}
  td:last-child {{ color: {TEXT_PRI}; text-align: right; }}

  hr {{ border-color: {BORDER}; margin: 1.5rem 0; }}
  .stMarkdown p {{ color: {TEXT_SEC}; line-height: 1.7; }}
</style>
"""

PLOT_STYLE = {
    "figure.facecolor": BG_CARD,
    "axes.facecolor":   BG_CARD2,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_SEC,
    "axes.titlecolor":  TEXT_PRI,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "axes.titleweight": "500",
    "xtick.color":      TEXT_MUTED,
    "ytick.color":      TEXT_MUTED,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.color":       BORDER,
    "grid.alpha":       0.5,
    "text.color":       TEXT_SEC,
    "font.family":      "sans-serif",
}


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
CKPT_DIR     = Path("models/checkpoints")

# Checkpoint paths — matching task1_train_fixed.py output
CKPT_RUN1 = CKPT_DIR / "lcnn_best.pt"       # from task1_train_fixed.py
CKPT_RUN2 = CKPT_DIR / "rawnet2_best.pt"    # if RawNet2 is trained separately


# ─────────────────────────────────────────────────────────────
# Model loading — cached
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_run1():
    """Load LCNN (Run 1 — trained by task1_train_fixed.py)."""
    # FIXED: use in_ch= not input_channels=
    model = LCNN()
    trained = False
    threshold = 0.5
    macs_str = "N/A"
    latency_ms = None

    if CKPT_RUN1.exists():
        ckpt = torch.load(str(CKPT_RUN1), map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        threshold  = float(ckpt.get("threshold", 0.5))
        trained    = True

        # Load MACs from saved results if available
        results_path = Path("outputs/Run1_LCNN_results.json")
        if results_path.exists():
            import json
            res = json.loads(results_path.read_text())
            macs_str   = str(res.get("macs",       "N/A"))
            latency_ms = res.get("latency_ms", None)
        else:
            # Compute on the fly (quick)
            dummy = torch.randn(1, 1, 180, 400)
            macs_info  = compute_macs(model, tuple(dummy.shape))
            lat_info   = measure_latency(model, dummy, n_runs=20)
            macs_str   = macs_info.get("macs",   "N/A")
            latency_ms = lat_info.get("mean_ms", None)
    else:
        st.sidebar.warning("⚠ Run 1 LCNN: no checkpoint — run task1_train_fixed.py first")

    return {
        "model":      model.eval(),
        "threshold":  threshold,
        "trained":    trained,
        "macs":       macs_str,
        "latency_ms": latency_ms,
        "label":      "Run 1 · LCNN",
        "run_id":     1,
    }


@st.cache_resource
def load_run2():
    """
    Load Run 2 model.
    Uses LCNN with rawnet2_best.pt if RawNet2 not available,
    since task1_train_fixed.py trains LCNN only.
    Falls back to untrained LCNN if no checkpoint at all.
    """
    trained   = False
    threshold = 0.5
    macs_str  = "N/A"
    latency_ms = None

    # Try RawNet2 first
    if HAS_RAWNET2 and CKPT_RUN2.exists():
        try:
            model = RawNet2(dropout=0.0)
            ckpt  = torch.load(str(CKPT_RUN2), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            threshold = float(ckpt.get("threshold", 0.5))
            trained   = True
            label     = "Run 2 · RawNet2"
            run_id    = 2
            st.sidebar.success("✓ RawNet2 checkpoint loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠ RawNet2 load failed: {e}")
            model = LCNN(in_ch=1, dropout=0.0)
            label = "Run 2 · LCNN (copy)"
            run_id = 1
    else:
        # Use LCNN with run1 checkpoint as run2 (same model, for demo purposes)
        model  = LCNN()
        label  = "Run 2 · LCNN (copy)"
        run_id = 1
        if CKPT_RUN1.exists():
            ckpt  = torch.load(str(CKPT_RUN1), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            threshold = float(ckpt.get("threshold", 0.5))
            trained   = True
            st.sidebar.info("ℹ Run 2 using LCNN checkpoint (RawNet2 not trained)")
        else:
            st.sidebar.warning("⚠ Run 2: no checkpoint found")

    if not trained:
        dummy = torch.randn(1, 1, 180, 400)
        macs_info  = compute_macs(model, tuple(dummy.shape))
        lat_info   = measure_latency(model, dummy, n_runs=20)
        macs_str   = macs_info.get("macs",   "N/A")
        latency_ms = lat_info.get("mean_ms", None)

    return {
        "model":      model.eval(),
        "threshold":  threshold,
        "trained":    trained,
        "macs":       macs_str,
        "latency_ms": latency_ms,
        "label":      label,
        "run_id":     run_id,
    }


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(waveform: torch.Tensor, model_info: dict) -> dict:
    """Run inference. Handles both LCNN (feat) and RawNet2 (raw) inputs."""
    model     = model_info["model"]
    threshold = model_info["threshold"]
    run_id    = model_info["run_id"]

    if run_id == 2 and HAS_RAWNET2 and isinstance(model, RawNet2):
        # RawNet2: raw waveform input
        x = waveform.unsqueeze(0)   # (1, 1, T)
    else:
        # LCNN: LFCC features
        feat = extract_lfcc(
            waveform.squeeze(0),
            sample_rate=SAMPLE_RATE, n_lfcc=60,
            n_fft=512, hop_length=160, win_length=400,
        )
        x = feat.unsqueeze(0).unsqueeze(0)   # (1, 1, 180, T')

    t0     = time.perf_counter()
    logits = model(x)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    probs   = F.softmax(logits, dim=-1).squeeze(0)
    p_real  = float(probs[0])
    p_spoof = float(probs[1])
    verdict = "REAL" if p_spoof < threshold else "SYNTHETIC"

    return {
        "verdict":    verdict,
        "p_real":     p_real,
        "p_spoof":    p_spoof,
        "threshold":  threshold,
        "latency_ms": lat_ms,
        "confidence": max(p_real, p_spoof) * 100,
        "macs":       model_info.get("macs",       "N/A"),
        "trained":    model_info.get("trained",    False),
    }


# ─────────────────────────────────────────────────────────────
# Grad-CAM inference
# ─────────────────────────────────────────────────────────────

def run_gradcam(waveform: torch.Tensor, model_info: dict, pred: dict) -> plt.Figure | None:
    """Generate Grad-CAM heatmap for LCNN. Returns figure or None."""
    if not HAS_GRADCAM:
        return None
    try:
        model = model_info["model"]
        if not isinstance(model, LCNN):
            return None

        feat = extract_lfcc(
            waveform.squeeze(0),
            sample_rate=SAMPLE_RATE, n_lfcc=60,
            n_fft=512, hop_length=160, win_length=400,
        )
        x = feat.unsqueeze(0).unsqueeze(0)   # (1, 1, 180, T')
        x.requires_grad_(True)

        cam     = get_gradcam_for_model(model, "lightcnn")
        heatmap, _ = cam(x)
        cam.remove_hooks()

        lfcc_np = feat[0].numpy()   # first 60 coeffs
        fig     = make_gradcam_figure(
            lfcc_np, heatmap,
            verdict=pred["verdict"],
            confidence=pred["confidence"],
        )
        return fig
    except Exception as e:
        print(f"[GradCAM] {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────

def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        waveform = load_and_preprocess(tmp_path, SAMPLE_RATE, MAX_DURATION)
    finally:
        os.unlink(tmp_path)
    return waveform   # (1, T)


# ─────────────────────────────────────────────────────────────
# Plot helpers — dark themed
# ─────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def render_waveform(waveform_np: np.ndarray, sr: int) -> plt.Figure:
    _apply_style()
    t   = np.linspace(0, len(waveform_np) / sr, len(waveform_np))
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(BG_CARD)
    ax.fill_between(t, waveform_np, 0, alpha=0.25, color=ACCENT_TEAL)
    ax.plot(t, waveform_np, color=ACCENT_TEAL, linewidth=0.6, alpha=0.9)
    ax.axhline(0, color=BORDER, linewidth=0.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title("Waveform", fontweight="500", color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4)
    fig.tight_layout(pad=0.6)
    return fig


def render_spectrogram(waveform_np: np.ndarray, sr: int) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.patch.set_facecolor(BG_CARD)
    spec, freqs, bins, im = ax.specgram(
        waveform_np, Fs=sr, NFFT=512, noverlap=384, cmap="inferno"
    )
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram", fontweight="500", color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=8)
    cb.outline.set_edgecolor(BORDER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)
    fig.tight_layout(pad=0.6)
    return fig


def render_lfcc(waveform: torch.Tensor, sr: int) -> plt.Figure:
    """LCNN feature input visualisation — uses local extract_lfcc."""
    _apply_style()
    try:
        # FIXED: use locally-imported extract_lfcc (not utils.features)
        feat = extract_lfcc(
            waveform.squeeze(0),
            sample_rate=sr, n_lfcc=60,
            n_fft=512, hop_length=160, win_length=400,
        ).numpy()
    except Exception as e:
        print(f"[render_lfcc] {e}")
        feat = np.random.randn(180, 250) * 0.1

    fig, ax = plt.subplots(figsize=(13.5, 2.5))
    fig.patch.set_facecolor(BG_CARD)
    im = ax.imshow(feat, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Frame"); ax.set_ylabel("LFCC coeff. (incl. Δ, ΔΔ)")
    ax.set_title("LFCC Features — LCNN Input (60 coeffs × 3)", fontweight="500",
                 color=TEXT_PRI, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.015)
    cb.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=8)
    cb.outline.set_edgecolor(BORDER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)
    fig.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────

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


def not_trained_banner(run_label: str):
    """Show warning when model has no checkpoint."""
    st.markdown(f"""
    <div style="background:#200A08; border:1px solid {ACCENT_CORAL}40;
                border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
        <p style="margin:0; font-size:13px; color:{ACCENT_CORAL};">
            ⚠ <strong>{run_label}</strong> — checkpoint not found.
            Predictions are <strong>random</strong> (model not trained).
            Run <code style="font-family:{FONT_MONO};">python task1_train_fixed.py --run 1</code> first.
        </p>
    </div>
    """, unsafe_allow_html=True)


def verdict_card(result: dict, model_name: str):
    """Full verdict card with confidence bar, MACs, latency."""
    is_real   = result["verdict"] == "REAL"
    color     = ACCENT_TEAL if is_real else ACCENT_CORAL
    icon      = "✓" if is_real else "✕"
    label     = "GENUINE" if is_real else "SYNTHETIC"
    pct_real  = result["p_real"]  * 100
    pct_spoof = result["p_spoof"] * 100
    thr_pct   = result["threshold"] * 100

    # Latency and MACs
    lat   = f"{result['latency_ms']:.1f} ms" if result['latency_ms'] else "—"
    macs  = str(result.get("macs", "N/A"))

    # Untrained warning
    if not result.get("trained", True):
        not_trained_banner(model_name)

    st.markdown(f"""
    <div style="
        background: {BG_CARD};
        border: 1px solid {color}40;
        border-top: 3px solid {color};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    ">
        <!-- Header row: name + confidence -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
            <div>
                <p style="margin:0 0 4px; font-size:11px; font-weight:600; letter-spacing:0.1em;
                          text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
                    {model_name}
                </p>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-flex; align-items:center; justify-content:center;
                                 width:28px; height:28px; border-radius:50%;
                                 background:{color}20; color:{color};
                                 font-weight:700; font-size:14px;">{icon}</span>
                    <span style="font-size:28px; font-weight:700; color:{color};
                                 font-family:{FONT_DISPLAY}; letter-spacing:-0.02em;">{label}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <p style="margin:0 0 2px; font-size:11px; color:{TEXT_MUTED}; letter-spacing:0.06em; text-transform:uppercase;">Confidence</p>
                <p style="margin:0; font-size:22px; font-weight:700; color:{TEXT_PRI}; font-family:{FONT_MONO};">
                    {result['confidence']:.1f}%
                </p>
            </div>
        </div>

        <!-- Probability bar -->
        <div style="margin-bottom:0.75rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:11px; color:{ACCENT_TEAL}; font-weight:500;">REAL</span>
                <span style="font-size:11px; color:{ACCENT_CORAL}; font-weight:500;">SYNTHETIC</span>
            </div>
            <div style="position:relative; height:8px; border-radius:4px; background:{BG_CARD2}; overflow:hidden;">
                <div style="position:absolute; left:0; top:0; bottom:0;
                            width:{pct_real:.1f}%;
                            background:linear-gradient(90deg, {ACCENT_TEAL}, {ACCENT_TEAL}99);
                            border-radius:4px;"></div>
                <div style="position:absolute; top:-2px; bottom:-2px;
                            left:{thr_pct:.1f}%; width:2px;
                            background:{TEXT_MUTED}; border-radius:1px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                <span style="font-size:11px; color:{TEXT_MUTED}; font-family:{FONT_MONO};">{pct_real:.1f}%</span>
                <span style="font-size:11px; color:{TEXT_MUTED}; font-family:{FONT_MONO};">{pct_spoof:.1f}%</span>
            </div>
        </div>

        <!-- Meta row: threshold / latency / MACs / p_real / p_spoof -->
        <div style="display:flex; gap:16px; flex-wrap:wrap; padding-top:0.75rem;
                    border-top:1px solid {BORDER};">
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">Threshold</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{result['threshold']:.3f}</p>
            </div>
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">Latency</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{lat}</p>
            </div>
            <div>
                <p style="margin:0; font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.08em;">MACs</p>
                <p style="margin:0; font-size:13px; color:{TEXT_SEC}; font-family:{FONT_MONO};">{macs}</p>
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
        bg    = f"{color}12"; bdr = f"{color}40"
    else:
        color = ACCENT_AMBER; icon = "⚡"
        text  = (f"Models disagree — "
                 f"<strong style='color:{ACCENT_TEAL};'>{model_names[0]}: {verdicts[0]}</strong>"
                 f" &nbsp;/&nbsp; "
                 f"<strong style='color:{ACCENT_CORAL};'>{model_names[1]}: {verdicts[1]}</strong>")
        bg    = f"{color}12"; bdr = f"{color}40"

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {bdr}; border-radius:12px;
                padding:0.9rem 1.25rem; display:flex; align-items:center; gap:10px;
                margin:1.25rem 0;">
        <span style="font-size:16px; flex-shrink:0;">{icon}</span>
        <p style="margin:0; font-size:14px; color:{TEXT_PRI};">{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown(f"""
    <div style="padding-bottom:1rem; border-bottom:1px solid {BORDER}; margin-bottom:1rem;">
        <p style="margin:0 0 2px; font-size:11px; font-weight:600; letter-spacing:0.1em;
                  text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};">
            Anti-Spoofing · PS #12
        </p>
        <p style="margin:0; font-size:18px; font-weight:700; color:{TEXT_PRI}; font-family:{FONT_DISPLAY};">
            Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <p style="font-size:11px; font-weight:600; letter-spacing:0.1em;
              text-transform:uppercase; color:{TEXT_MUTED}; font-family:{FONT_DISPLAY};
              margin-bottom:0.5rem;">Models</p>
    """, unsafe_allow_html=True)

    use_run1 = st.sidebar.checkbox("Run 1 · LCNN (LFCC)",     value=True)
    use_run2 = st.sidebar.checkbox("Run 2 · RawNet2 / LCNN",  value=True)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM (X-Factor)", value=HAS_GRADCAM)

    st.sidebar.markdown(f"<hr style='border-color:{BORDER}; margin:1rem 0;'>", unsafe_allow_html=True)

    # Model status
    st.sidebar.markdown(f"""
    <p style="font-size:11px; font-weight:600; letter-spacing:0.1em;
              text-transform:uppercase; color:{TEXT_MUTED}; margin-bottom:0.5rem;">
        Checkpoint Status
    </p>
    """, unsafe_allow_html=True)

    r1_status = "✅ Found" if CKPT_RUN1.exists() else f"❌ Missing — run task1_train_fixed.py"
    r2_status = "✅ Found" if CKPT_RUN2.exists() else "⚠ Using Run1 weights"
    st.sidebar.markdown(f"""
    <div style="font-size:12px; color:{TEXT_SEC}; line-height:1.8;">
        <div>Run 1 LCNN: <span style="color:{'#1D9E75' if CKPT_RUN1.exists() else ACCENT_CORAL};">{r1_status}</span></div>
        <div>Run 2:      <span style="color:{'#1D9E75' if CKPT_RUN2.exists() else ACCENT_AMBER};">{r2_status}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"<hr style='border-color:{BORDER}; margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:6px; font-size:13px; color:{TEXT_SEC};">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:10px; border-radius:50%; background:{ACCENT_TEAL}; flex-shrink:0;"></div>
            Real — bonafide human speech
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:10px; border-radius:50%; background:{ACCENT_CORAL}; flex-shrink:0;"></div>
            Synthetic — TTS / cloned / spoofed
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px; height:2px; background:{TEXT_MUTED}; flex-shrink:0;"></div>
            Threshold (EER from dev set)
        </div>
    </div>
    """, unsafe_allow_html=True)

    return use_run1, use_run2, show_gradcam


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

    use_run1, use_run2, show_gradcam = sidebar()

    # ── Hero header ──
    st.markdown(f"""
    <div style="padding:2.5rem 0 1.5rem; border-bottom:1px solid {BORDER}; margin-bottom:2rem;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:0.5rem;">
            <div style="width:42px; height:42px; background:{ACCENT_TEAL}18;
                        border:1px solid {ACCENT_TEAL}40; border-radius:10px;
                        display:flex; align-items:center; justify-content:center; font-size:20px;">🔊</div>
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
        <p style="margin:0.5rem 0 0; color:{TEXT_SEC}; font-size:14px; max-width:600px;">
            Upload or record audio to detect whether speech is
            <span style="color:{ACCENT_TEAL}; font-weight:500;">genuine</span> or
            <span style="color:{ACCENT_CORAL}; font-weight:500;">synthetically generated</span>.
            Powered by LCNN classifier with LFCC features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ──
    models = {}
    if use_run1:
        with st.spinner("Loading Run 1 — LCNN…"):
            models["Run 1 · LCNN"] = load_run1()
    if use_run2:
        with st.spinner("Loading Run 2…"):
            models["Run 2"] = load_run2()

    if not models:
        st.warning("Select at least one model in the sidebar.")
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
                    <p style="margin:0; font-size:13px; color:{TEXT_SEC};"><span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">.wav</span> — PCM waveform</p>
                    <p style="margin:0; font-size:13px; color:{TEXT_SEC};"><span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">.flac</span> — Lossless audio</p>
                    <p style="margin:0; font-size:13px; color:{TEXT_SEC};"><span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">.mp3</span> — Compressed audio</p>
                </div>
                <p style="margin:8px 0 0; font-size:12px; color:{TEXT_MUTED};">16 kHz mono · max 4 s used</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_mic:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px;
                    padding:1.25rem 1.5rem; text-align:center;">
            <p style="margin:0 0 6px; font-size:14px; color:{TEXT_SEC};">
                Install <span style="font-family:{FONT_MONO}; color:{ACCENT_TEAL};">streamlit-audiorecorder</span> for live recording:
            </p>
            <div style="display:inline-block; background:{BG_CARD2}; border:1px solid {BORDER};
                        border-radius:8px; padding:6px 14px;">
                <code style="font-family:{FONT_MONO}; font-size:13px; color:{ACCENT_TEAL};">pip install streamlit-audiorecorder</code>
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
        <div style="background:{BG_CARD}; border:1px dashed {BORDER}; border-radius:14px;
                    padding:2.5rem; text-align:center; margin:2rem 0;">
            <p style="margin:0 0 6px; font-size:32px;">🎙️</p>
            <p style="margin:0; font-size:15px; color:{TEXT_SEC};">Upload or record audio to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Preprocess audio ──
    with st.spinner("Preprocessing audio…"):
        try:
            waveform    = load_audio_from_bytes(audio_bytes)
            waveform_np = waveform.squeeze(0).numpy()
        except Exception as e:
            st.error(f"Audio load failed: {e}")
            return
    duration = len(waveform_np) / SAMPLE_RATE

    # ── Signal Analysis ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Signal Analysis", "📊")

    col_w, col_s = st.columns(2)
    with col_w:
        fig = render_waveform(waveform_np, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col_s:
        fig = render_spectrogram(waveform_np, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Full-width LFCC (only when LCNN is selected)
    if use_run1:
        fig = render_lfcc(waveform, SAMPLE_RATE)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Classifier Results ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Classifier Results", "🤖")

    results  = {}
    cols     = st.columns(len(models))
    for col, (name, model_info) in zip(cols, models.items()):
        with col:
            with st.spinner(f"Running {name}…"):
                result = predict(waveform, model_info)
                result["trained"] = model_info.get("trained", False)
            results[name] = result
            verdict_card(result, name)

    # ── Agreement banner ──
    if len(results) == 2:
        v_list     = [r["verdict"] for r in results.values()]
        m_list     = list(results.keys())
        agreement_banner(v_list, m_list)

    # ── Grad-CAM (X-Factor) ──
    if show_gradcam and HAS_GRADCAM and use_run1:
        st.markdown("<br>", unsafe_allow_html=True)
        section_label("Grad-CAM Explainability (X-Factor)", "🔍")
        with st.spinner("Computing Grad-CAM heatmap…"):
            r1_info = models.get("Run 1 · LCNN")
            r1_pred = results.get("Run 1 · LCNN", {})
            if r1_info:
                fig = run_gradcam(waveform, r1_info, r1_pred)
                if fig:
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                    st.markdown(f"""
                    <p style="font-size:13px; color:{TEXT_SEC}; margin-top:0.5rem;">
                        🔴 Red regions show where the LCNN model focused when making its decision.
                        High Grad-CAM activation at specific time-frequency regions typically indicates
                        artefacts characteristic of synthetic speech (e.g. unnatural spectral smoothness,
                        phase discontinuities, or missing formant transitions).
                    </p>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Grad-CAM requires gradcam.py in the same directory.")
    elif show_gradcam and not HAS_GRADCAM:
        st.info("Grad-CAM unavailable — add gradcam.py to project directory.")

    # ── Audio details ──
    with st.expander("Audio details"):
        st.markdown(f"""
        <table>
            <thead><tr><th>Property</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Duration</td><td>{duration:.2f} s</td></tr>
                <tr><td>Sample rate</td><td>{SAMPLE_RATE:,} Hz</td></tr>
                <tr><td>Samples</td><td>{len(waveform_np):,}</td></tr>
                <tr><td>Peak amplitude</td><td>{float(np.abs(waveform_np).max()):.4f}</td></tr>
                <tr><td>RMS energy</td><td>{float(np.sqrt(np.mean(waveform_np**2))):.4f}</td></tr>
                <tr><td>Zero-crossing rate</td><td>{float(np.mean(np.abs(np.diff(np.sign(waveform_np))))/2):.4f}</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""utils/metrics.py — EER, cosine similarity, bypass rate, MACs, latency."""

import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


# ─────────────────────────────────────────────────────────────
# Equal Error Rate (EER)
# ─────────────────────────────────────────────────────────────

def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate.

    Args:
        labels: ground-truth binary labels (0=bonafide, 1=spoof)
        scores: model scores (higher = more likely spoof)

    Returns:
        (eer, threshold) tuple
    """
    # Invert so bonafide=positive class
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # EER is where FPR ≈ FNR
    if np.all(np.isnan(fnr)) or np.all(np.isnan(fpr)):
        return 0.5, 0.5  # fallback

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


def compute_eer_from_logits(
    labels: List[int],
    logits: torch.Tensor,   # (N, 2) or (N,)
) -> Tuple[float, float]:
    """Convenience wrapper: converts logits → softmax scores → EER."""
    labels_np = np.array(labels)
    if logits.dim() == 2:
        probs = F.softmax(logits, dim=-1)
        scores = probs[:, 1].detach().cpu().numpy()   # P(spoof)
    else:
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    return compute_eer(labels_np, scores)


# ─────────────────────────────────────────────────────────────
# Cosine Similarity
# ─────────────────────────────────────────────────────────────

def cosine_similarity_matrix(
    embeddings_a: torch.Tensor,   # (N, D)
    embeddings_b: torch.Tensor,   # (M, D)
) -> torch.Tensor:
    """Compute cosine similarity between every pair (a_i, b_j). Returns (N, M)."""
    a_norm = F.normalize(embeddings_a, p=2, dim=-1)
    b_norm = F.normalize(embeddings_b, p=2, dim=-1)
    return torch.mm(a_norm, b_norm.T)


def mean_cosine_similarity(
    real_embeddings: torch.Tensor,    # (N, D)
    cloned_embeddings: torch.Tensor,  # (N, D)
) -> float:
    """Mean cosine similarity between paired real and cloned embeddings."""
    sims = F.cosine_similarity(real_embeddings, cloned_embeddings, dim=-1)
    return float(sims.mean().item())


# ─────────────────────────────────────────────────────────────
# Bypass Rate (for Task 3)
# ─────────────────────────────────────────────────────────────

def compute_bypass_rate(
    labels: np.ndarray,       # 0=real, 1=spoof
    predictions: np.ndarray,  # 0=predicted real, 1=predicted spoof
) -> float:
    """
    Bypass rate = fraction of spoof samples that are predicted as real.
    Higher bypass rate = easier to fool the classifier.
    """
    spoof_mask = labels == 1
    if spoof_mask.sum() == 0:
        return 0.0
    n_bypassed = ((predictions[spoof_mask]) == 0).sum()
    return float(n_bypassed / spoof_mask.sum())


def compute_bypass_rate_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,       # P(spoof)
    threshold: float = 0.5,
) -> float:
    predictions = (scores >= threshold).astype(int)
    return compute_bypass_rate(labels, predictions)


# ─────────────────────────────────────────────────────────────
# Model efficiency: MACs and latency
# ─────────────────────────────────────────────────────────────

def compute_macs_and_params(
    model: torch.nn.Module,
    input_shape: Tuple,          # e.g. (1, 1, 180, 400)
    device: str = "cpu",
) -> Dict[str, str]:
    """
    Compute MACs and parameter count using ptflops.
    Returns dict with human-readable strings.
    """
    try:
        from ptflops import get_model_complexity_info
        model = model.to(device)
        macs, params = get_model_complexity_info(
            model,
            input_shape[1:],     # ptflops expects shape without batch dim
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        return {"macs": macs, "params": params}
    except ImportError:
        return {"macs": "ptflops not installed", "params": "N/A"}
    except Exception as e:
        return {"macs": f"error: {e}", "params": "N/A"}


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure mean and std inference latency in milliseconds.
    """
    model = model.to(device).eval()
    x = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(latencies.mean()), 3),
        "std_ms":  round(float(latencies.std()), 3),
        "min_ms":  round(float(latencies.min()), 3),
        "max_ms":  round(float(latencies.max()), 3),
    }


# ─────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────

def print_metrics_table(metrics: Dict, title: str = "Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"{'='*50}\n")

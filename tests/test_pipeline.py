"""tests/test_pipeline.py — Quick sanity checks (no dataset needed).

Run: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio


def test_lcnn():
    from models.lcnn import LCNN
    model = LCNN()
    x = torch.randn(2, 1, 180, 400)
    out = model(x)
    assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
    emb = model.get_embedding(x)
    assert emb.shape == (2, 160), f"Expected (2,160), got {emb.shape}"
    print("  ✓ LCNN forward pass OK")


def test_rawnet2():
    from models.rawnet2 import RawNet2
    model = RawNet2()
    x = torch.randn(2, 1, 64000)
    out = model(x)
    assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
    print("  ✓ RawNet2 forward pass OK")


def test_lfcc():
    from utils.features import extract_lfcc
    wav = torch.randn(64000)
    feat = extract_lfcc(wav, sample_rate=16000, n_lfcc=60)
    assert feat.shape[0] == 180, f"Expected 180 features (60*3), got {feat.shape[0]}"
    print(f"  ✓ LFCC extraction OK — shape: {feat.shape}")


def test_mel():
    from utils.features import extract_mel
    wav = torch.randn(64000)
    feat = extract_mel(wav, sample_rate=16000, n_mels=80)
    assert feat.shape[0] == 80
    print(f"  ✓ Mel extraction OK — shape: {feat.shape}")


def test_eer():
    from utils.metrics import compute_eer
    np.random.seed(42)
    labels = np.array([0]*100 + [1]*100)
    scores = np.concatenate([
        np.random.normal(0.3, 0.1, 100),
        np.random.normal(0.7, 0.1, 100),
    ])
    eer, thr = compute_eer(labels, scores)
    assert 0.0 <= eer <= 1.0
    print(f"  ✓ EER computation OK — EER={eer*100:.2f}%, threshold={thr:.3f}")


def test_noise():
    from utils.noise import add_noise
    signal = np.random.randn(16000).astype(np.float32)
    for noise_type in ["awgn", "babble", "music"]:
        noisy = add_noise(signal, noise_type, snr_db=10, sample_rate=16000)
        assert noisy.shape == signal.shape
        assert not np.any(np.isnan(noisy))
    print("  ✓ Noise injection OK (AWGN, babble, music)")


def test_cosine_similarity():
    from utils.metrics import mean_cosine_similarity
    a = torch.randn(10, 256)
    b = torch.randn(10, 256)
    sim = mean_cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0
    # Identity similarity
    sim_self = mean_cosine_similarity(a, a)
    assert abs(sim_self - 1.0) < 1e-5
    print(f"  ✓ Cosine similarity OK — random pair sim={sim:.4f}")


def test_bypass_rate():
    from utils.metrics import compute_bypass_rate
    labels = np.array([1]*100)
    preds  = np.array([0]*30 + [1]*70)   # 30 bypassed
    rate = compute_bypass_rate(labels, preds)
    assert abs(rate - 0.30) < 1e-6
    print(f"  ✓ Bypass rate OK — {rate*100:.0f}%")


def test_macs():
    from models.lcnn   import LCNN
    from utils.metrics import compute_macs_and_params
    model = LCNN()
    info = compute_macs_and_params(model, (1, 1, 180, 400))
    print(f"  ✓ MACs computation OK — {info}")


if __name__ == "__main__":
    print("Running pipeline sanity checks...\n")
    tests = [
        test_lfcc,
        test_mel,
        test_eer,
        test_noise,
        test_cosine_similarity,
        test_bypass_rate,
        test_lcnn,
        test_rawnet2,
        test_macs,
    ]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"  Failed: {failed}/{len(tests)}")
    print(f"{'='*40}")
    sys.exit(0 if failed == 0 else 1)

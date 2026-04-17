# PS #12 Group24 — Spoofing Your Own Anti-Spoofing System

A complete pipeline for voice anti-spoofing classification and voice cloning attacks.

## Project Structure

```
voice-antispoofing-arena/
├── app.py                      # Streamlit demo
├── dataset.py                  # Custom dataset (real vs cloned)
├── features.py                 # LFCC / Mel extraction
├── metrics.py                  # EER, cosine similarity, bypass rate
├── noise.py                    # Noise injection (AWGN, babble, music)
├── plots.py                    # Visualization utilities
├── lcnn.py                     # Run 1: LCNN classifier
├── rawnet2.py                  # Run 2: RawNet2 classifier
├── speaker_encoder.py          # Speaker similarity (optional)

├── task0_voice_clone.py        # Task 0: Voice cloning
├── task1_train.py              # Task 1: Train classifier
├── task2_attack.py             # Task 2: Attack + similarity
├── task3_noise_robustness.py   # Task 3: Noise stress testing
├── run_all.py                  # Run full pipeline

├── test_pipeline.py        # Sanity checks (9/9 passed)
│

├── data/
│   ├── reference_speaker/      # Real audio
│   ├── cloned_audio/           # Generated audio

├── outputs/
│   ├── plots/
│   ├── task2_results.json
│   ├── task3_results.json

├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create environment
conda create -n antispoof python=3.10 -y
conda activate antispoof

# 2. Install dependencies
pip install -r requirements.txt
```

## Running Tasks

```bash
# Task 0 — Voice cloning
python task0_voice_clone.py

# Task 1 — Train classifier (LCNN)
python task1_train.py

# Task 2 — Attack analysis
python task2_attack.py

# Task 3 — Noise robustness
python task3_noise_robustness.py

# Full pipeline
python run_all.py
```

## Evaluation Metrics

## Evaluation Metrics

| Task | Metric | Description |
|------|--------|-------------|
| 1 | EER ↓ | Equal Error Rate — lower indicates better discrimination |
| 1 | Loss curves | Training/validation BCE loss over epochs |
| 2 | Cosine similarity | Speaker embedding similarity (real vs cloned) |
| 2 | Attack EER ↑ | Higher value indicates increased vulnerability to spoofing |
| 3 | Bypass rate ↓ | % of spoofed audio misclassified as real |
| 3 | SNR trend | Bypass rate variation across noise levels (0–20 dB) |
| Tech | MACs | Model computational complexity |
| Tech | Latency | Inference time per sample |


## Data

Place your audio files in:

- `data/reference_speaker/` → real recordings  
- `data/cloned_audio/` → generated cloned audio

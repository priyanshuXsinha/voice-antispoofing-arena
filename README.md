# PS #12 — Spoofing Your Own Anti-Spoofing System

A complete pipeline for voice anti-spoofing classification and voice cloning attacks.

## Project Structure

```
antispoof_project_dl_Group24/
├── configs/
│   └── config.yaml              # Hyperparameters and paths
├── data/
│   ├── reference_speaker/       # Real audio (your recordings)
│   ├── cloned_audio/            # Generated cloned speech
├── models/
│   ├── lcnn.py                  # Run 1: LCNN classifier
│   ├── rawnet2.py               # Run 2: RawNet2 classifier
│   └── speaker_encoder.py       # Speaker similarity (optional)
├── utils/
│   ├── dataset.py               # Custom real vs cloned dataset
│   ├── features.py              # LFCC / Mel extraction
│   ├── metrics.py               # EER, cosine similarity, bypass rate
│   ├── noise.py                 # Noise injection (AWGN, babble, music)
│   └── plots.py                 # Visualization utilities
├── scripts/
│   ├── task0_voice_clone.py     # Task 0: Voice cloning
│   ├── task1_train.py           # Task 1: Train classifier
│   ├── task2_attack.py          # Task 2: Attack + similarity
│   ├── task3_noise_robustness.py# Task 3: Noise stress testing
│   └── run_all.py               # Run full pipeline
├── demo/
│   └── app.py                   # Streamlit demo (optional)
├── tests/
│   └── test_pipeline.py         # Sanity checks (9/9 passed)
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

# 3. Download ASVspoof2019 LA (see data/download_asvspoof.md)

# 4. Edit configs/config.yaml to set your data paths
```

## Running Tasks

```bash
# Task 0 — Voice cloning
python scripts/task0_voice_clone.py

# Task 1 — Train classifiers (Run 1: LCNN, Run 2: RawNet2)
python scripts/task1_train.py --run 1   # LCNN
python scripts/task1_train.py --run 2   # RawNet2

# Task 2 — Attack analysis
python scripts/task2_attack.py

# Task 3 — Noise robustness
python scripts/task3_noise_robustness.py

# Task 4 — Demo
streamlit run demo/app.py

# Full pipeline
python scripts/run_all.py
```

## Evaluation Metrics

| Task | Metric | Description |
|------|--------|-------------|
| 1 | EER ↓ | Equal Error Rate — lower is better |
| 1 | Loss curves | Train/val BCE loss over epochs |
| 2 | Cosine sim | Real vs synthetic speaker embedding similarity |
| 2 | Attack EER ↑ | Higher = easier to fool classifier |
| 3 | Bypass rate | % spoofed audio classified as real under noise |
| 3 | SNR trend | Bypass rate vs SNR (0–20 dB) |
| Tech | MACs | Multiply-accumulate ops (model efficiency) |
| Tech | Latency | Inference time (ms) |

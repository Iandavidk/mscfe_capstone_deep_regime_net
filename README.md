# Deep-RegimeNet
### An Unsupervised Deep Clustering Framework for Adaptive Market Regime Identification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Group 16019 | WorldQuant University MScFE 690 Capstone Project**  
Ian Kihara Wangui · Grace Nmon Monanyun · Immaculate Wanjiru Kimani

---

## Overview

Financial markets shift between distinct behavioural regimes — bull, bear, and
high-volatility states — with direct implications for portfolio construction and
risk management. Classical methods (Hidden Markov Models, Markov Switching AR)
rely on parametric assumptions and sequential separation of feature learning
from cluster assignment.

**Deep-RegimeNet** jointly optimises temporal representation learning and market
regime cluster assignment in a single end-to-end neural architecture, combining:

- A **Temporal Autoencoder** (LSTM with temporal attention pooling + LayerNorm)
- A **Deep Embedded Clustering (DEC)** module with Student's t-distribution kernel
- A **composite loss** L = L_rec + λ · L_clust with dynamic lambda annealing

Four baselines are implemented for rigorous comparison:
1. DTW-KMeans
2. Sequential AE + KMeans
3. Gaussian HMM
4. Naïve Volatility Quantile classifier

---

## Repository Structure

```
mscfe_capstone_deep_regime_net/
├── v3_1_draft_report_capstone_deep_regime_net.ipynb   # Main notebook
├── config.py                                           # Hyperparameters & seeds
├── requirements.txt                                    # Python dependencies
├── CONTRIBUTORS.md                                     # Team roles
├── LICENSE                                             # MIT License
├── README.md                                           # This file
└── previous_notebooks/                                 # Earlier versions
```

---

## Requirements

Python 3.9 or higher. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Experiments were run on **Google Colab T4 GPU**. Training Deep-RegimeNet on CPU
is possible but approximately 10× slower. For local GPU use, ensure CUDA 12 is installed.

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open the notebook in Colab
2. Set Runtime → Change runtime type → T4 GPU
3. Run all cells in order (Runtime → Run all)

### Option 2 — Local Jupyter

```bash
git clone https://github.com/Iandavidk/mscfe_capstone_deep_regime_net.git
cd mscfe_capstone_deep_regime_net
pip install -r requirements.txt
jupyter notebook v3_1_draft_report_capstone_deep_regime_net.ipynb
```

---

## Reproducing Results

All random seeds are controlled in `config.py`. The primary results use:

| Parameter          | Value                            |
|--------------------|----------------------------------|
| Primary seed       | 42                               |
| Multi-seed runs    | [42, 123, 456, 789, 2024]        |
| Final architecture | 2 layers, 256 hidden, 64 latent, window=20 |
| λ                  | 0.1                              |
| K (regimes)        | 4                                |
| Data               | S&P 500 (^GSPC), 1993–2026      |

Expected key outputs:

| Metric                                    | Value   |
|-------------------------------------------|---------|
| DeepRegimeNet Annualised Sharpe           | 0.284   |
| Gaussian HMM Annualised Sharpe            | 0.850   |
| Naïve Volatility Quantile Sharpe          | 0.425   |
| DeepRegimeNet mean concordance purity     | 58.9%   |
| Joint vs Sequential Silhouette Δ          | −16.5%  |

---

## Data

Data is downloaded automatically via `yfinance` — no manual download required.
The notebook fetches S&P 500 daily OHLCV from Yahoo Finance (ticker: ^GSPC,
auto-adjusted, 1993-01-01 to 2026-01-01). Four features are engineered:

| Feature              | Description                          | Normalisation         |
|----------------------|--------------------------------------|-----------------------|
| `log_return`         | Daily log return                     | Rolling Z-score 252d  |
| `intraday_range`     | (High − Low) / Close                 | Rolling Z-score 252d  |
| `volume_z`           | Volume Z-score (20-day rolling)      | Rolling Z-score 20d   |
| `log_realised_vol`   | Log 20-day annualised realised vol   | Training-fold stats   |

---

## Methodology Summary

| Phase     | Description                                                              |
|-----------|--------------------------------------------------------------------------|
| Phase I   | TAE pre-training on reconstruction loss only (80 epochs)                 |
| Phase II  | Centroid initialisation via K-Means on pre-trained latent codes          |
| Phase III | Joint training with dynamic lambda annealing (150 epochs)                |

Validation uses 5-fold walk-forward cross-validation with a 60-day embargo at
each boundary. λ and K are selected by reconstruction loss on a held-out 15%
validation split, decoupled from clustering evaluation metrics.

---

## Key Results Summary

The main finding is that Deep-RegimeNet does **not** outperform simpler baselines
on economic endpoints at this sample size. The Gaussian HMM achieved the highest
Annualised Sharpe (0.850) and Total Return (1,816.995%), while the Naïve Volatility
Quantile classifier outperformed Deep-RegimeNet on all strategy metrics without
any training or GPU computation. Deep-RegimeNet achieved the highest historical
concordance purity (58.9%) across seven pre-specified market event windows.

This constitutes a methodologically rigorous negative result with value for the
literature: it demonstrates that DEC-based deep temporal clustering does not
overcome its known instabilities (cluster collapse, initialisation sensitivity,
lack of temporal coherence regularisation) at financial daily-data sample sizes.

---

## Citation

If you use this code, please cite:

```
Wangui, I. K., Monanyun, G. N., & Kimani, I. W. (2026). Deep-RegimeNet:
An Unsupervised Deep Clustering Framework for Adaptive Market Regime
Identification. MScFE 690 Capstone Project, WorldQuant University.
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

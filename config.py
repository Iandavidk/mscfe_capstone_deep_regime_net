"""
config.py — Deep-RegimeNet hyperparameter configuration
========================================================
Group 16019 | MScFE 690 Capstone | WorldQuant University | 2026

All tunable parameters are centralised here so the notebook can be
reproduced without manually editing code cells.

Usage in notebook (replace Cell 4 hardcoded values with):
    from config import *
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED  = 42
MULTI_SEEDS  = [42, 123, 456, 789, 2024]

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER       = "^GSPC"          # S&P 500 index
DATA_START   = "1993-01-01"
DATA_END     = "2026-01-01"

# ── Preprocessing ─────────────────────────────────────────────────────────────
ROLLING_WINDOW = 252            # trading days for rolling Z-score normalisation
VOL_WINDOW     = 20             # days for realised volatility calculation

# ── Walk-forward cross-validation ─────────────────────────────────────────────
TRAIN_YEARS    = 10
VAL_YEARS      = 5
TEST_YEARS     = 3
EMBARGO_DAYS   = 60             # days embargoed at each fold boundary

# ── Sliding windows ───────────────────────────────────────────────────────────
WINDOW_SIZE    = 20             # trading days per window (final best)
STRIDE         = 1              # fully overlapping

# ── Model architecture (final best from grid search) ──────────────────────────
BEST_NUM_LAYERS  = 2
BEST_HIDDEN_DIM  = 256
BEST_LATENT_DIM  = 64
BEST_DROPOUT     = 0.1
BEST_RNN         = "LSTM"       # LSTM outperformed GRU in Ablation A

# ── Clustering (final best from λ/K selection via reconstruction loss) ────────
BEST_K           = 4            # number of regimes
BEST_LAM         = 0.1          # clustering loss weight (λ)
ALPHA            = 1.0          # Student-t degrees of freedom (Cauchy kernel)

# ── Training protocol ─────────────────────────────────────────────────────────
PRETRAIN_EPOCHS  = 80           # Phase I: reconstruction only
JOINT_EPOCHS     = 150          # Phase III: joint optimisation
ANNEAL_EPOCHS    = 75           # λ ramps from 0 → BEST_LAM over this many epochs
TARGET_UPDATE    = 5            # recompute P every N joint epochs
LR_PRETRAIN      = 1e-3         # Phase I learning rate
LR_JOINT         = 5e-4         # Phase III learning rate
WEIGHT_DECAY     = 1e-5
GRAD_CLIP        = 1.0          # gradient clipping norm
BATCH_SIZE       = 256

# ── Ablation study grids ──────────────────────────────────────────────────────
ABLATION_LAM_GRID  = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
ABLATION_K_GRID    = [2, 3, 4, 5, 6]
SELECTION_LAM      = [0.01, 0.05, 0.1, 0.2]   # λ/K final selection grid
SELECTION_K        = [3, 4, 5, 6]
GRID_SAMPLE_N      = 30         # random configs drawn from 243 total candidates

# ── Evaluation ────────────────────────────────────────────────────────────────
TRADING_DAYS       = 252
MIN_REGIME_OBS     = 10         # minimum observations to report a regime

# ── Known historical event windows (pre-specified before any training) ────────
KNOWN_EVENTS = {
    "Dot-com Crash":       ("2000-03-01", "2002-10-01"),
    "GFC":                 ("2007-10-01", "2009-03-01"),
    "Eurozone Crisis":     ("2011-07-01", "2011-10-01"),
    "2015-16 Selloff":     ("2015-08-01", "2016-02-01"),
    "Covid Crash":         ("2020-02-01", "2020-03-31"),
    "Post-GFC Bull":       ("2009-03-01", "2020-02-01"),
    "Post-Covid Bull":     ("2020-03-01", "2022-01-01"),
}

# SOL/USDT Forecasting Research System (Mac M2, lightweight)

A local **research** system for forecasting SOL/USDT on **1h candles**.

It includes:
- Direction classifiers for **4h, 12h, 24h** (`up/down`)
- Volatility expansion classifier for **24h**
- Models: **Logistic Regression**, **XGBoost**, plus **KMeans regime classifier**
- Walk-forward backtest with leakage-safe time splits
- Metrics, artifacts, prediction logs
- Streamlit dashboard

## 1) Project structure

```
.
├── artifacts/models
├── configs
├── data/raw
├── data/processed
├── notebooks
├── reports
├── src
│   ├── backtest
│   ├── dashboard
│   ├── data
│   ├── features
│   ├── models
│   └── utils
├── environment.yml
└── Makefile
```

## 2) Prerequisites

- macOS (Apple Silicon/M2 is fine)
- Conda installed (Miniconda or Anaconda)
- Terminal access

If conda is not installed, install Miniconda first.

## 3) Setup (copy/paste)

From this folder:

```bash
cd /Users/lucatorrisi/Documents/GitHub/solana-predictor
make setup
```

This creates/updates conda env: `solana-research`.

## 4) Run pipeline (copy/paste)

```bash
make fetch
make build-features
make train
make backtest
```

## 5) Launch dashboard

```bash
make dashboard
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## 6) What each step does

- `make fetch`
  - Downloads SOL/USDT 1h OHLCV via Binance/ccxt
  - Saves to `data/raw/sol_1h.csv`
  - Incremental updates supported

- `make build-features`
  - Computes returns, RSI, MACD, Bollinger, ATR, ADX, rolling vol, volume z-score, trend strength
  - Builds targets:
    - `y_dir_4h`, `y_dir_12h`, `y_dir_24h`
    - `y_vol_24h` (future vol expansion)
  - Saves:
    - `data/processed/dataset_1h.parquet`
    - `data/processed/dataset_1h.csv`

- `make train`
  - Trains KMeans regime classifier (4 regimes)
  - Trains final Logistic + XGBoost models for each target
  - Saves to `artifacts/models/`

- `make backtest`
  - Walk-forward time-series backtest (no leakage)
  - Saves:
    - `reports/metrics.json`
    - `reports/metrics.csv`
    - `reports/predictions.csv`
    - `reports/calibration.csv`

## 7) Metrics reported

Per target/model:
- Accuracy
- Precision
- Recall
- ROC-AUC
- Brier score (calibration quality)
- Confusion matrix
- Strategy expectancy estimate after fees

Prediction log columns:
- `timestamp`
- `horizon`
- `model_name`
- `proba_up`
- `predicted_label`
- `actual_label`
- `regime`

## 8) Notes

- This is **research only** (not live trading).
- Designed for low memory usage:
  - 1h candles only
  - no deep learning
  - compact model set
- If exchange data fetch fails, the fetch script can optionally generate synthetic fallback data so the pipeline remains runnable for testing.

## 9) Optional direct commands (without Make)

```bash
conda run -n solana-research python -m src.data.fetch_ohlcv --config configs/default.yaml
conda run -n solana-research python -m src.features.build_features --config configs/default.yaml
conda run -n solana-research python -m src.models.train_models --config configs/default.yaml
conda run -n solana-research python -m src.backtest.walkforward --config configs/default.yaml
conda run -n solana-research streamlit run src/dashboard/app.py -- --config configs/default.yaml
```

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _forward_realized_vol(log_returns: pd.Series, horizon: int) -> pd.Series:
    values = log_returns.to_numpy(dtype=float)
    out = np.full_like(values, fill_value=np.nan)
    n = len(values)
    for i in range(n - horizon):
        window = values[i + 1 : i + 1 + horizon]
        out[i] = float(np.nanstd(window, ddof=0))
    return pd.Series(out, index=log_returns.index)


def build_features(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    raw_path = Path(data_cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data: {raw_path}. Run fetch step first.")

    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Raw data is empty: {raw_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["return_1h"] = close.pct_change()
    df["log_return_1h"] = np.log(close).diff()

    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bb_pband"] = bb.bollinger_pband()
    df["bb_wband"] = bb.bollinger_wband()

    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df["adx_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()

    rolling_vol_window = int(feat_cfg.get("rolling_vol_window", 24))
    df[f"rolling_vol_{rolling_vol_window}"] = df["log_return_1h"].rolling(rolling_vol_window).std()

    volume_z_window = int(feat_cfg.get("volume_z_window", 24))
    vol_mean = volume.rolling(volume_z_window).mean()
    vol_std = volume.rolling(volume_z_window).std()
    df["volume_zscore_24"] = (volume - vol_mean) / vol_std

    sma_12 = close.rolling(12).mean()
    sma_48 = close.rolling(48).mean()
    df["trend_strength"] = (sma_12 - sma_48).abs() / close

    # Targets
    for h in feat_cfg.get("target_horizons", [4, 12, 24]):
        h_int = int(h)
        future_close = close.shift(-h_int)
        df[f"future_return_{h_int}h"] = (future_close / close) - 1.0
        df[f"y_dir_{h_int}h"] = (future_close > close).astype(float)

    vol_h = int(feat_cfg.get("vol_horizon", 24))
    current_vol_col = f"rolling_vol_{rolling_vol_window}"
    df[f"future_realized_vol_{vol_h}h"] = _forward_realized_vol(df["log_return_1h"], horizon=vol_h)

    lookback = int(feat_cfg.get("vol_threshold_lookback", 2160))
    q = float(feat_cfg.get("vol_threshold_quantile", 0.75))
    df["vol_threshold"] = df[current_vol_col].rolling(lookback, min_periods=100).quantile(q)
    df[f"y_vol_{vol_h}h"] = (df[f"future_realized_vol_{vol_h}h"] > df["vol_threshold"]).astype(float)

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1h",
        "log_return_1h",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_hband",
        "bb_lband",
        "bb_pband",
        "bb_wband",
        "atr_14",
        "adx_14",
        f"rolling_vol_{rolling_vol_window}",
        "volume_zscore_24",
        "trend_strength",
    ]
    target_cols = [f"y_dir_{int(h)}h" for h in feat_cfg.get("target_horizons", [4, 12, 24])] + [f"y_vol_{vol_h}h"]

    keep_cols = ["timestamp"] + feature_cols + target_cols + [
        f"future_return_{int(h)}h" for h in feat_cfg.get("target_horizons", [4, 12, 24])
    ]
    dataset = df[keep_cols].dropna().reset_index(drop=True)

    parquet_path = Path(data_cfg["processed_parquet_path"])
    csv_path = Path(data_cfg["processed_csv_path"])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(parquet_path, index=False)
    dataset.to_csv(csv_path, index=False)

    LOGGER.info("Feature dataset rows=%d cols=%d -> %s", dataset.shape[0], dataset.shape[1], parquet_path)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature set and targets.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    build_features(cfg)


if __name__ == "__main__":
    main()

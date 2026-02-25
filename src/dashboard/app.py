from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_known_args()[0]


def _load_csv(path: str | Path, required: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing file: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    st.set_page_config(page_title="SOL Research Dashboard", layout="wide")
    st.title("SOL/USDT Forecasting Research Dashboard")

    raw = _load_csv(cfg["data"]["raw_path"])
    preds = _load_csv(cfg["artifacts"]["predictions_csv"], required=False)
    metrics = _load_csv(cfg["artifacts"]["metrics_csv"], required=False)

    regime_path = Path(cfg["data"]["dataset_with_regime_path"])
    regime_df = pd.read_parquet(regime_path) if regime_path.exists() else pd.DataFrame()

    if raw.empty:
        st.error("No market data found. Run: make fetch")
        st.stop()

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Latest SOL/USDT 1h Price")
        fig = px.line(raw.tail(500), x="timestamp", y="close", labels={"close": "Price"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Current State")
        latest_price = float(raw.iloc[-1]["close"])
        st.metric("Latest Price", f"{latest_price:.4f}")

        if not regime_df.empty and "regime" in regime_df.columns:
            regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"], utc=True)
            latest_regime = int(regime_df.sort_values("timestamp").iloc[-1]["regime"])
            st.metric("Latest Regime", str(latest_regime))
        else:
            st.info("Regime unavailable. Run: make train")

    st.subheader("Latest Model Probabilities")
    if preds.empty:
        st.warning("No predictions found yet. Run: make backtest")
    else:
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
        latest_ts = preds["timestamp"].max()
        latest = preds[preds["timestamp"] == latest_ts].copy()
        latest = latest.sort_values(["horizon", "model_name"]) 

        cols = st.columns(max(1, len(latest)))
        for idx, (_, row) in enumerate(latest.iterrows()):
            with cols[idx % len(cols)]:
                st.caption(f"{row['horizon']} • {row['model_name']}")
                p_up = float(row["proba_up"])
                st.metric("P(up)", f"{p_up:.2%}")
                st.progress(float(np.clip(p_up, 0.0, 1.0)))

    st.subheader("Backtest Metrics")
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True)
    else:
        st.info("Metrics unavailable. Run: make backtest")

    st.subheader("Cumulative Edge Proxy (XGB vs Logistic Baseline)")
    if not preds.empty:
        probs = preds.copy()
        pivot = probs.pivot_table(
            index=["timestamp", "horizon", "actual_label"],
            columns="model_name",
            values="proba_up",
            aggfunc="mean",
        ).reset_index()

        if {"xgb", "logreg"}.issubset(set(pivot.columns)):
            eps = 1e-9
            y = pivot["actual_label"].astype(int)
            p_xgb = np.clip(pivot["xgb"], eps, 1 - eps)
            p_lr = np.clip(pivot["logreg"], eps, 1 - eps)
            logloss_xgb = -(y * np.log(p_xgb) + (1 - y) * np.log(1 - p_xgb))
            logloss_lr = -(y * np.log(p_lr) + (1 - y) * np.log(1 - p_lr))
            pivot["edge"] = logloss_lr - logloss_xgb
            pivot = pivot.sort_values("timestamp")
            pivot["cum_edge"] = pivot.groupby("horizon")["edge"].cumsum()

            fig_edge = px.line(pivot, x="timestamp", y="cum_edge", color="horizon")
            st.plotly_chart(fig_edge, use_container_width=True)
        else:
            st.info("Need both xgb and logreg predictions for edge chart.")


if __name__ == "__main__":
    main()

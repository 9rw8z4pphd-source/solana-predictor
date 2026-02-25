from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import ccxt  # type: ignore
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _generate_synthetic(start_ts: pd.Timestamp, periods: int) -> pd.DataFrame:
    """Create synthetic 1h OHLCV fallback data if exchange fetch fails."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start=start_ts, periods=periods, freq="1h", tz="UTC")
    log_rets = rng.normal(0, 0.02, size=periods)
    close = 100 * np.exp(np.cumsum(log_rets))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1 - rng.normal(0, 0.002))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.001, 0.002, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.001, 0.002, periods)))
    volume = np.exp(rng.normal(12.0, 0.5, periods))
    out = pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return out


def fetch_incremental(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw_path = Path(data_cfg["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(raw_path)
    min_history_hours = int(data_cfg.get("min_history_hours", 3000))
    since_ms: int | None
    if existing.empty:
        since_ms = int((pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=min_history_hours)).timestamp() * 1000)
    else:
        since_ms = int(existing["timestamp"].max().timestamp() * 1000) + 1

    ex_name = str(data_cfg.get("exchange", "binance"))
    symbol = str(data_cfg.get("symbol", "SOL/USDT"))
    timeframe = str(data_cfg.get("timeframe", "1h"))
    limit = int(data_cfg.get("ohlcv_limit", 1000))

    LOGGER.info("Fetching %s %s from %s since=%s", symbol, timeframe, ex_name, since_ms)

    try:
        exchange = getattr(ccxt, ex_name)({"enableRateLimit": True})
        all_rows: list[list[float]] = []
        now_ms = exchange.milliseconds()
        fetch_since = since_ms

        while fetch_since is not None and fetch_since < now_ms:
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = int(rows[-1][0])
            if last_ts <= fetch_since:
                break
            fetch_since = last_ts + 1
            now_ms = exchange.milliseconds()
            time.sleep(exchange.rateLimit / 1000)

        fetched = pd.DataFrame(all_rows, columns=COLUMNS)
        if fetched.empty:
            LOGGER.warning("No new rows fetched from exchange.")
            merged = existing.copy()
        else:
            fetched["timestamp"] = pd.to_datetime(fetched["timestamp"], unit="ms", utc=True)
            merged = pd.concat([existing, fetched], ignore_index=True)
    except Exception as exc:  # pragma: no cover - fallback path
        if not bool(data_cfg.get("allow_synthetic_fallback", True)):
            raise RuntimeError(f"Failed to fetch exchange data and fallback disabled: {exc}") from exc
        LOGGER.warning("Exchange fetch failed (%s). Using synthetic fallback data.", exc)
        start = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=min_history_hours)
        synthetic = _generate_synthetic(start, min_history_hours)
        merged = pd.concat([existing, synthetic], ignore_index=True)

    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(raw_path, index=False)
    LOGGER.info("Saved %d rows -> %s", len(merged), raw_path)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch/update SOL/USDT 1h OHLCV data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    fetch_incremental(cfg)


if __name__ == "__main__":
    main()

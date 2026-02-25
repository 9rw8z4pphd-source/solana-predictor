from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


FEATURE_EXCLUDE_PREFIXES = ("y_", "future_return_")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "timestamp" and not c.startswith(FEATURE_EXCLUDE_PREFIXES) and c != "regime"]


def _build_models(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(model_cfg["logistic"]["C"]),
                        max_iter=int(model_cfg["logistic"]["max_iter"]),
                        random_state=int(model_cfg["random_state"]),
                    ),
                ),
            ]
        ),
        "xgb": XGBClassifier(
            n_estimators=int(model_cfg["xgboost"]["n_estimators"]),
            max_depth=int(model_cfg["xgboost"]["max_depth"]),
            learning_rate=float(model_cfg["xgboost"]["learning_rate"]),
            subsample=float(model_cfg["xgboost"]["subsample"]),
            colsample_bytree=float(model_cfg["xgboost"]["colsample_bytree"]),
            reg_lambda=float(model_cfg["xgboost"]["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(model_cfg["random_state"]),
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _h_from_target(target: str) -> int:
    # y_dir_4h -> 4, y_vol_24h -> 24
    return int(target.split("_")[-1].replace("h", ""))


def _expectancy(df_pred: pd.DataFrame, long_t: float, short_t: float, fee_per_side: float, horizon_col: str) -> dict[str, float]:
    proba = df_pred["proba_up"]
    future_ret = df_pred[horizon_col]
    pos = np.where(proba >= long_t, 1, np.where(proba <= short_t, -1, 0))
    gross = pos * future_ret
    fees = np.where(pos != 0, fee_per_side * 2, 0.0)
    net = gross - fees
    traded = pos != 0
    if traded.sum() == 0:
        return {"expectancy": 0.0, "trade_count": 0.0, "win_rate": 0.0}
    return {
        "expectancy": float(net[traded].mean()),
        "trade_count": float(traded.sum()),
        "win_rate": float((net[traded] > 0).mean()),
    }


def run_backtest(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    feat_cfg = cfg["features"]
    bt_cfg = cfg["backtest"]
    artifacts_cfg = cfg["artifacts"]

    dataset_path = Path(data_cfg["dataset_with_regime_path"])
    if not dataset_path.exists():
        dataset_path = Path(data_cfg["processed_parquet_path"])

    if not dataset_path.exists():
        raise FileNotFoundError("Missing dataset. Run build-features and train first.")

    df = pd.read_parquet(dataset_path).sort_values("timestamp").reset_index(drop=True)
    if "regime" not in df.columns:
        regime_bundle = joblib.load(Path(artifacts_cfg["model_dir"]) / "regime_kmeans.joblib")
        X_reg = regime_bundle["scaler"].transform(df[regime_bundle["features"]])
        df["regime"] = regime_bundle["model"].predict(X_reg)

    feature_cols = _feature_columns(df)
    target_cols = [f"y_dir_{int(h)}h" for h in feat_cfg["target_horizons"]] + [f"y_vol_{int(feat_cfg['vol_horizon'])}h"]

    tscv = TimeSeriesSplit(
        n_splits=int(model_cfg["n_splits"]),
        test_size=int(model_cfg.get("tscv_test_size", 300)),
    )

    pred_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for target in target_cols:
        horizon = _h_from_target(target)
        horizon_ret_col = f"future_return_{horizon}h"

        for model_name, model in _build_models(model_cfg).items():
            fold_probs: list[np.ndarray] = []
            fold_preds: list[np.ndarray] = []
            fold_actuals: list[np.ndarray] = []
            fold_idx: list[np.ndarray] = []

            for train_idx, test_idx in tscv.split(df):
                X_train = df.iloc[train_idx][feature_cols].to_numpy()
                y_train = df.iloc[train_idx][target].astype(int).to_numpy()
                X_test = df.iloc[test_idx][feature_cols].to_numpy()
                y_test = df.iloc[test_idx][target].astype(int).to_numpy()

                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)

                fold_probs.append(proba)
                fold_preds.append(pred)
                fold_actuals.append(y_test)
                fold_idx.append(test_idx)

            probs = np.concatenate(fold_probs)
            preds = np.concatenate(fold_preds)
            actuals = np.concatenate(fold_actuals)
            idx_all = np.concatenate(fold_idx)

            sub = df.iloc[idx_all].copy().reset_index(drop=True)
            sub_out = pd.DataFrame(
                {
                    "timestamp": sub["timestamp"],
                    "horizon": target,
                    "model_name": model_name,
                    "proba_up": probs,
                    "predicted_label": preds,
                    "actual_label": actuals,
                    "regime": sub["regime"],
                    "future_return_h": sub[horizon_ret_col].to_numpy(),
                }
            )
            pred_rows.extend(sub_out.to_dict(orient="records"))

            acc = accuracy_score(actuals, preds)
            prec = precision_score(actuals, preds, zero_division=0)
            rec = recall_score(actuals, preds, zero_division=0)
            auc = roc_auc_score(actuals, probs) if len(np.unique(actuals)) > 1 else np.nan
            brier = brier_score_loss(actuals, probs)
            cm = confusion_matrix(actuals, preds, labels=[0, 1]).tolist()

            cal_true, cal_pred = calibration_curve(actuals, probs, n_bins=10, strategy="quantile")
            for b_idx, (pt, pp) in enumerate(zip(cal_true, cal_pred)):
                calibration_rows.append(
                    {
                        "horizon": target,
                        "model_name": model_name,
                        "bin": b_idx,
                        "prob_true": float(pt),
                        "prob_pred": float(pp),
                    }
                )

            exp = _expectancy(
                sub_out.rename(columns={"future_return_h": horizon_ret_col}),
                long_t=float(bt_cfg["long_threshold"]),
                short_t=float(bt_cfg["short_threshold"]),
                fee_per_side=float(bt_cfg["fee_per_side"]),
                horizon_col=horizon_ret_col,
            )

            metric_rows.append(
                {
                    "horizon": target,
                    "model_name": model_name,
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "roc_auc": float(auc) if not np.isnan(auc) else None,
                    "brier_score": float(brier),
                    "confusion_matrix": cm,
                    "expectancy_after_fees": exp["expectancy"],
                    "trade_count": exp["trade_count"],
                    "win_rate": exp["win_rate"],
                }
            )

    predictions_df = pd.DataFrame(pred_rows).sort_values(["timestamp", "horizon", "model_name"])
    metrics_df = pd.DataFrame(metric_rows).sort_values(["horizon", "model_name"])
    calibration_df = pd.DataFrame(calibration_rows)

    pred_out = Path(artifacts_cfg["predictions_csv"])
    metrics_csv = Path(artifacts_cfg["metrics_csv"])
    metrics_json = Path(artifacts_cfg["metrics_json"])
    cal_out = Path(artifacts_cfg["calibration_csv"])
    for p in [pred_out, metrics_csv, metrics_json, cal_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(pred_out, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    calibration_df.to_csv(cal_out, index=False)

    metrics_json_data = {
        "rows": metric_rows,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_rows": int(len(df)),
    }
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_json_data, f, indent=2)

    LOGGER.info("Backtest complete. Metrics rows=%d, predictions rows=%d", len(metrics_df), len(predictions_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest with leakage-safe time splits.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    run_backtest(cfg)


if __name__ == "__main__":
    main()

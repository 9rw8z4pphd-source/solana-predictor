from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("y_", "future_return_")
    cols = [c for c in df.columns if c not in {"timestamp"} and not c.startswith(excluded_prefixes)]
    return cols


def train_models(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    regime_cfg = cfg["regime"]
    feat_cfg = cfg["features"]
    artifacts_cfg = cfg["artifacts"]

    dataset_path = Path(data_cfg["processed_parquet_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {dataset_path}. Run build-features first.")

    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    feature_cols = _get_feature_columns(df)
    model_dir = Path(artifacts_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Regime classifier
    regime_features = [c for c in regime_cfg["feature_columns"] if c in df.columns]
    if len(regime_features) < 3:
        raise ValueError(f"Regime feature columns missing. Needed: {regime_cfg['feature_columns']}")

    regime_scaler = StandardScaler()
    X_reg = regime_scaler.fit_transform(df[regime_features])
    kmeans = KMeans(n_clusters=int(regime_cfg["n_clusters"]), random_state=int(model_cfg["random_state"]), n_init=10)
    regimes = kmeans.fit_predict(X_reg)
    df_reg = df.copy()
    df_reg["regime"] = regimes

    out_regime_path = Path(data_cfg["dataset_with_regime_path"])
    df_reg.to_parquet(out_regime_path, index=False)

    joblib.dump({"scaler": regime_scaler, "model": kmeans, "features": regime_features}, model_dir / "regime_kmeans.joblib")

    target_cols = [f"y_dir_{int(h)}h" for h in feat_cfg["target_horizons"]] + [f"y_vol_{int(feat_cfg['vol_horizon'])}h"]

    manifest: dict[str, Any] = {
        "feature_columns": feature_cols,
        "regime_features": regime_features,
        "targets": target_cols,
        "models": {},
    }

    X = df[feature_cols].to_numpy()
    for target in target_cols:
        y = df[target].astype(int).to_numpy()

        lr = Pipeline(
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
        )
        lr.fit(X, y)
        lr_path = model_dir / f"{target}_logreg.joblib"
        joblib.dump(lr, lr_path)

        xgb = XGBClassifier(
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
        )
        xgb.fit(X, y)
        xgb_path = model_dir / f"{target}_xgb.joblib"
        joblib.dump(xgb, xgb_path)

        manifest["models"][target] = {
            "logreg": str(lr_path),
            "xgb": str(xgb_path),
        }

    with (model_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info("Trained and saved models for targets: %s", ", ".join(target_cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LR, XGB and regime models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    train_models(cfg)


if __name__ == "__main__":
    main()

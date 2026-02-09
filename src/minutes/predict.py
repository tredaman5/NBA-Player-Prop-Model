"""
predict.py (minutes)

Loads trained minutes model + feature list from models_artifacts/
and produces minutes predictions for a given minutes dataset.

Output:
- data/processed/minutes_with_pred_<season>_<season_type>.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _default_dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"minutes_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _model_paths(season: str, season_type: str) -> tuple[Path, Path]:
    model_path = ARTIFACTS_DIR / f"minutes_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"minutes_features_{season}_{season_type.replace(' ', '')}.json"
    return model_path, feats_path


def predict_minutes(
    season: str,
    season_type: str = "Regular Season",
    dataset_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads the minutes dataset and adds MIN_PRED column.

    Returns DataFrame with predictions and saves to parquet.
    """
    if dataset_path is None:
        dataset_path = _default_dataset_path(season, season_type)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_path, feats_path = _model_paths(season, season_type)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feats_path}")

    model = joblib.load(model_path)
    features = json.loads(feats_path.read_text())

    df = pd.read_parquet(dataset_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Ensure feature columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[features].astype(float)
    preds = model.predict(X)

    # Reasonable clipping (minutes can't be negative or > 60 in normal games)
    df["MIN_PRED"] = np.clip(preds, 0, 60)

    if out_path is None:
        out_path = DATA_PROCESSED / f"minutes_with_pred_{season}_{season_type.replace(' ', '')}.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Saved minutes predictions -> {out_path}")
    return df


if __name__ == "__main__":
    # Example: current season
    predict_minutes(season="2025-26", season_type="Regular Season")

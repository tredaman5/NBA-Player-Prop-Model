"""
predict.py (points props)

Loads points model + feature list from models_artifacts and produces predictions.

Outputs:
- data/processed/points_with_pred_<season>_<season_type>.parquet

Optional:
- If a line is provided, computes P(Over line) using a simple Normal approximation
  with sigma estimated from recent residuals stored in points_eval json (fallback to 4.0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]  # .../src/props/points/predict.py -> repo
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"points_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _artifact_paths(season: str, season_type: str) -> tuple[Path, Path, Path]:
    model_path = ARTIFACTS_DIR / f"points_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"points_features_{season}_{season_type.replace(' ', '')}.json"
    eval_path = ARTIFACTS_DIR / f"points_eval_{season}_{season_type.replace(' ', '')}.json"
    return model_path, feats_path, eval_path


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predict_points(
    season: str,
    season_type: str = "Regular Season",
    dataset_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    line: Optional[float] = None,
    sigma: Optional[float] = None,
) -> pd.DataFrame:
    """
    Adds:
    - PTS_PRED
    - If line provided: P_OVER_LINE

    sigma: std dev for Normal approximation. If None, use RMSE from eval file as a proxy.
    """
    if dataset_path is None:
        dataset_path = _dataset_path(season, season_type)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_path, feats_path, eval_path = _artifact_paths(season, season_type)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feats_path}")

    model = joblib.load(model_path)
    features = json.loads(feats_path.read_text())

    # Estimate sigma from eval RMSE (proxy) if not provided
    if sigma is None:
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            sigma = float(ev.get("rmse", 4.0))
        else:
            sigma = 4.0

    df = pd.read_parquet(dataset_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[features].astype(float)
    preds = model.predict(X)

    # Points can't be negative; cap at something sane
    df["PTS_PRED"] = np.clip(preds, 0, 80)

    if line is not None:
        # P(PTS > line) under Normal(mu=pred, sigma)
        # P(PTS > line) = 1 - Phi((line - mu)/sigma)
        z = (line - df["PTS_PRED"]) / float(sigma)
        df[f"P_OVER_{line}"] = 1.0 - z.apply(_normal_cdf)

    if out_path is None:
        out_path = DATA_PROCESSED / f"points_with_pred_{season}_{season_type.replace(' ', '')}.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Saved points predictions -> {out_path}")
    if line is not None:
        print(f"Computed P(over {line}) using sigma={sigma:.3f}")

    return df


if __name__ == "__main__":
    # Default: just produce predictions for the full dataset
    predict_points(season="2025-26", season_type="Regular Season")

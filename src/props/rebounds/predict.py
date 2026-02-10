"""
predict.py (rebounds props)

Loads rebounds model + feature list and produces predictions.

Output:
- data/processed/rebounds_with_pred_<season>_<season_type>.parquet

Optional:
- If a line is provided, computes P(Over line) using Normal approximation
  with sigma taken from eval RMSE (fallback to 2.0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"rebounds_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _artifact_paths(season: str, season_type: str) -> tuple[Path, Path, Path]:
    model_path = ARTIFACTS_DIR / f"rebounds_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"rebounds_features_{season}_{season_type.replace(' ', '')}.json"
    eval_path = ARTIFACTS_DIR / f"rebounds_eval_{season}_{season_type.replace(' ', '')}.json"
    return model_path, feats_path, eval_path


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predict_rebounds(
    season: str,
    season_type: str = "Regular Season",
    dataset_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    line: Optional[float] = None,
    sigma: Optional[float] = None,
) -> pd.DataFrame:
    if dataset_path is None:
        dataset_path = _dataset_path(season, season_type)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_path, feats_path, eval_path = _artifact_paths(season, season_type)
    model = joblib.load(model_path)
    features = json.loads(feats_path.read_text())

    if sigma is None:
        sigma = 2.0
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            sigma = float(ev.get("rmse", sigma))

    df = pd.read_parquet(dataset_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[features].astype(float)
    preds = model.predict(X)
    df["REB_PRED"] = np.clip(preds, 0, 40)

    if line is not None:
        z = (line - df["REB_PRED"]) / float(sigma)
        df[f"P_OVER_{line}"] = 1.0 - z.apply(_normal_cdf)

    if out_path is None:
        out_path = DATA_PROCESSED / f"rebounds_with_pred_{season}_{season_type.replace(' ', '')}.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Saved rebounds predictions -> {out_path}")
    if line is not None:
        print(f"Computed P(over {line}) using sigma={sigma:.3f}")

    return df


if __name__ == "__main__":
    predict_rebounds(season="2025-26", season_type="Regular Season")

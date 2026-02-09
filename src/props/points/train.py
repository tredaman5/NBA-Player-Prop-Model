"""
train.py (points props)

Trains a points prediction model using the processed points dataset.
Saves:
- models_artifacts/points_model_<season>_<season_type>.joblib
- models_artifacts/points_features_<season>_<season_type>.json
- models_artifacts/points_eval_<season>_<season_type>.json

Target: PTS
Primary feature: MIN_PRED + rolling opportunity features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[3]  # points/train.py -> props -> src -> repo
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"points_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("GAME_DATE").copy()
    cutoff_index = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff_index].copy(), df.iloc[cutoff_index:].copy()


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Select numeric features only; keep it consistent across runs.
    """
    candidates = [
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        "MIN_PRED",
        "FGA_PER_MIN_ROLL_3",
        "FGA_PER_MIN_ROLL_5",
        "FGA_PER_MIN_ROLL_10",
        "FTA_PER_MIN_ROLL_3",
        "FTA_PER_MIN_ROLL_5",
        "FTA_PER_MIN_ROLL_10",
        "FG3A_PER_MIN_ROLL_3",
        "FG3A_PER_MIN_ROLL_5",
        "FG3A_PER_MIN_ROLL_10",
        "PTS_PER_MIN_ROLL_3",
        "PTS_PER_MIN_ROLL_5",
        "PTS_PER_MIN_ROLL_10",
        "TEAM_PTS_ROLL_5",
        "TEAM_PTS_ROLL_10",
    ]
    return [c for c in candidates if c in df.columns]


def train_points_model(
    season: str,
    season_type: str = "Regular Season",
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> None:
    _ensure_dir(ARTIFACTS_DIR)

    path = _dataset_path(season, season_type)
    if not path.exists():
        raise FileNotFoundError(f"Points dataset not found: {path}\nRun: python -m src.props.points.build_dataset")

    df = pd.read_parquet(path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    df = df.dropna(subset=["PTS"]).copy()

    features = _select_features(df)
    if not features:
        raise ValueError("No features found. Check points dataset columns.")

    # Drop rows with missing key rolling features (early season)
    roll_feats = [c for c in features if "ROLL" in c or c == "MIN_PRED"]
    df = df.dropna(subset=roll_feats).copy()

    train_df, test_df = _time_split(df, test_ratio=test_ratio)

    X_train = train_df[features].astype(float)
    y_train = train_df["PTS"].astype(float)

    X_test = test_df[features].astype(float)
    y_test = test_df["PTS"].astype(float)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Baseline: MIN_PRED * (PTS_PER_MIN_ROLL_10 if available else PTS_PER_MIN_ROLL_5)
    ppm_col = "PTS_PER_MIN_ROLL_10" if "PTS_PER_MIN_ROLL_10" in features else (
        "PTS_PER_MIN_ROLL_5" if "PTS_PER_MIN_ROLL_5" in features else None
    )
    if ppm_col and "MIN_PRED" in features:
        baseline = X_test["MIN_PRED"].to_numpy() * X_test[ppm_col].to_numpy()
        b_mae = mean_absolute_error(y_test, baseline)
        b_rmse = mean_squared_error(y_test, baseline) ** 0.5
    else:
        b_mae, b_rmse = np.nan, np.nan

    print("\n=== Points Model Evaluation ===")
    print(f"Dataset: {path.name}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Features ({len(features)}): {features}")
    print(f"Model MAE:  {mae:.3f} points")
    print(f"Model RMSE: {rmse:.3f} points")
    print(f"Baseline MAE:  {b_mae:.3f} points")
    print(f"Baseline RMSE: {b_rmse:.3f} points")

    # Save artifacts
    model_path = ARTIFACTS_DIR / f"points_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"points_features_{season}_{season_type.replace(' ', '')}.json"
    eval_path = ARTIFACTS_DIR / f"points_eval_{season}_{season_type.replace(' ', '')}.json"

    joblib.dump(model, model_path)
    feats_path.write_text(json.dumps(features, indent=2))

    eval_obj = {
        "season": season,
        "season_type": season_type,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "features": features,
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_mae": float(b_mae) if np.isfinite(b_mae) else None,
        "baseline_rmse": float(b_rmse) if np.isfinite(b_rmse) else None,
        "test_date_min": str(test_df["GAME_DATE"].min().date()),
        "test_date_max": str(test_df["GAME_DATE"].max().date()),
    }
    eval_path.write_text(json.dumps(eval_obj, indent=2))

    print("\nSaved:")
    print(f"- {model_path}")
    print(f"- {feats_path}")
    print(f"- {eval_path}")


if __name__ == "__main__":
    train_points_model(season="2025-26", season_type="Regular Season")

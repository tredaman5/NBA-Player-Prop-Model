"""
train.py (minutes)

Trains a minutes prediction model from the processed minutes dataset.
Saves:
- models_artifacts/minutes_model_<season>_<season_type>.joblib
- models_artifacts/minutes_features_<season>_<season_type>.json

Target: MIN
Features: rolling minutes, rest, home/away + light boxscore context (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _default_dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"minutes_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: last X% of dates is test.
    Avoids leakage across time.
    """
    df = df.sort_values("GAME_DATE").copy()
    cutoff_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:cutoff_index].copy()
    test_df = df.iloc[cutoff_index:].copy()
    return train_df, test_df


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Keep features that should exist from your dataset builder.
    Only include numeric / boolean features here.
    """
    candidates = [
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        # optional context columns if present
        "PTS",
        "REB",
        "AST",
        "FGA",
        "FTA",
        "FG3A",
        "TOV",
    ]
    return [c for c in candidates if c in df.columns]


def train_minutes_model(
    season: str,
    season_type: str = "Regular Season",
    dataset_path: Path | None = None,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Load dataset, train model, evaluate, and save artifacts.
    """
    _ensure_dir(ARTIFACTS_DIR)

    if dataset_path is None:
        dataset_path = _default_dataset_path(season, season_type)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    # Ensure types
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.dropna(subset=["MIN"]).copy()

    features = _select_features(df)

    if not features:
        raise ValueError("No features found. Check your dataset columns.")

    # Drop rows where key rolling features are missing (rare early-season)
    df = df.dropna(subset=[f for f in features if f.startswith("MIN_ROLL_")]).copy()

    # Time split
    train_df, test_df = _time_split(df, test_ratio=test_ratio)

    X_train = train_df[features].astype(float)
    y_train = train_df["MIN"].astype(float)

    X_test = test_df[features].astype(float)
    y_test = test_df["MIN"].astype(float)

    # Model: strong baseline, fast, handles nonlinearities
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Simple baseline: MIN_ROLL_5 as prediction
    if "MIN_ROLL_5" in features:
        baseline = X_test["MIN_ROLL_5"].to_numpy()
        b_mae = mean_absolute_error(y_test, baseline)
        b_rmse = mean_squared_error(y_test, baseline) ** 0.5
    else:
        b_mae, b_rmse = np.nan, np.nan

    print("\n=== Minutes Model Evaluation ===")
    print(f"Dataset: {dataset_path.name}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Features ({len(features)}): {features}")
    print(f"Model MAE:  {mae:.3f} minutes")
    print(f"Model RMSE: {rmse:.3f} minutes")
    print(f"Baseline (MIN_ROLL_5) MAE:  {b_mae:.3f} minutes")
    print(f"Baseline (MIN_ROLL_5) RMSE: {b_rmse:.3f} minutes")

    # Save artifacts
    model_path = ARTIFACTS_DIR / f"minutes_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"minutes_features_{season}_{season_type.replace(' ', '')}.json"

    joblib.dump(model, model_path)
    feats_path.write_text(json.dumps(features, indent=2))

    print("\nSaved:")
    print(f"- {model_path}")
    print(f"- {feats_path}")


if __name__ == "__main__":
    # Example: train on current season 2025-26
    train_minutes_model(season="2025-26", season_type="Regular Season")

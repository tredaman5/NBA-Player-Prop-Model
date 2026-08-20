"""
train.py (assists props)

Trains an assists prediction model.
Saves:
- models_artifacts/assists_model_<season>_<season_type>.joblib
- models_artifacts/assists_features_<season>_<season_type>.json
- models_artifacts/assists_eval_<season>_<season_type>.json
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


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dataset_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"assists_dataset_{season}_{season_type.replace(' ', '')}.parquet"


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("GAME_DATE").copy()
    cutoff = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def _select_features(df: pd.DataFrame) -> List[str]:
    candidates = [
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        "MIN_PRED",
        "AST_PER_MIN_ROLL_3",
        "AST_PER_MIN_ROLL_5",
        "AST_PER_MIN_ROLL_10",
        "TEAM_PTS_ROLL_5",
        "TEAM_PTS_ROLL_10",
        "OPP_AST_ALLOWED_ROLL_5",
        "OPP_AST_ALLOWED_ROLL_10",
    ]
    return [c for c in candidates if c in df.columns]


def train_assists_model(
    season: str,
    season_type: str = "Regular Season",
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> None:
    _ensure_dir(ARTIFACTS_DIR)

    path = _dataset_path(season, season_type)
    if not path.exists():
        raise FileNotFoundError(f"Assists dataset not found: {path}\nRun: python -m src.props.assists.build_dataset")

    df = pd.read_parquet(path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
    df = df.dropna(subset=["AST"]).copy()

    features = _select_features(df)
    if not features:
        raise ValueError("No features found. Check assists dataset columns.")

    df = df.dropna(subset=[c for c in features if "ROLL" in c or c == "MIN_PRED"]).copy()

    train_df, test_df = _time_split(df, test_ratio=test_ratio)

    X_train = train_df[features].astype(float)
    y_train = train_df["AST"].astype(float)
    X_test = test_df[features].astype(float)
    y_test = test_df["AST"].astype(float)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 30)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Baseline: MIN_PRED * AST_PER_MIN_ROLL_10 or _5
    apm_col = "AST_PER_MIN_ROLL_10" if "AST_PER_MIN_ROLL_10" in features else (
        "AST_PER_MIN_ROLL_5" if "AST_PER_MIN_ROLL_5" in features else None
    )
    if apm_col and "MIN_PRED" in features:
        baseline = X_test["MIN_PRED"].to_numpy() * X_test[apm_col].to_numpy()
        b_mae = mean_absolute_error(y_test, baseline)
        b_rmse = mean_squared_error(y_test, baseline) ** 0.5
    else:
        b_mae, b_rmse = np.nan, np.nan

    print("\n=== Assists Model Evaluation ===")
    print(f"Dataset: {path.name}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Features ({len(features)}): {features}")
    print(f"Model MAE:  {mae:.3f} assists")
    print(f"Model RMSE: {rmse:.3f} assists")
    print(f"Baseline MAE:  {b_mae:.3f} assists")
    print(f"Baseline RMSE: {b_rmse:.3f} assists")

    model_path = ARTIFACTS_DIR / f"assists_model_{season}_{season_type.replace(' ', '')}.joblib"
    feats_path = ARTIFACTS_DIR / f"assists_features_{season}_{season_type.replace(' ', '')}.json"
    eval_path = ARTIFACTS_DIR / f"assists_eval_{season}_{season_type.replace(' ', '')}.json"

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
    train_assists_model(season="2025-26", season_type="Regular Season")

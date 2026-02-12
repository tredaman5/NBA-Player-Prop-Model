"""
predict_slate.py

Predicts props for a slate date from slate feature tables.

Inputs:
- data/processed/slate_<prop>_features_<date>.parquet

Uses artifacts:
- models_artifacts/<prop>_model_<season>_<seasontype>.joblib
- models_artifacts/<prop>_features_<season>_<seasontype>.json
- models_artifacts/<prop>_eval_<season>_<seasontype>.json (for sigma/RMSE)

Outputs:
- data/processed/slate_<prop>_preds_<date>.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "models_artifacts"


def _st(season_type: str) -> str:
    return season_type.replace(" ", "")


def _slate_features_path(prop: str, date_str: str) -> Path:
    return PROCESSED / f"slate_{prop}_features_{date_str}.parquet"


def _slate_preds_path(prop: str, date_str: str) -> Path:
    return PROCESSED / f"slate_{prop}_preds_{date_str}.parquet"


def _artifact_paths(prop: str, season: str, season_type: str) -> Dict[str, Path]:
    st = _st(season_type)
    return {
        "model": ARTIFACTS / f"{prop}_model_{season}_{st}.joblib",
        "features": ARTIFACTS / f"{prop}_features_{season}_{st}.json",
        "eval": ARTIFACTS / f"{prop}_eval_{season}_{st}.json",
    }


def _pred_col(prop: str) -> str:
    return {"points": "PTS_PRED", "rebounds": "REB_PRED", "assists": "AST_PRED"}[prop]


def _sigma_default(prop: str) -> float:
    return {"points": 4.0, "rebounds": 2.0, "assists": 1.7}[prop]


def predict_slate(prop: str, date_str: str, season: str, season_type: str) -> Path:
    feat_path = _slate_features_path(prop, date_str)
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing slate features: {feat_path}\nRun: python -m src.slate.build_slate --date {date_str}")

    paths = _artifact_paths(prop, season, season_type)
    if not paths["model"].exists() or not paths["features"].exists():
        raise FileNotFoundError(
            f"Missing model artifacts for {prop}.\n"
            f"Expected:\n  {paths['model']}\n  {paths['features']}\n"
            f"Run: python main.py train-prop --prop {prop} --season {season}"
        )

    model = joblib.load(paths["model"])
    features = json.loads(paths["features"].read_text())

    sigma = _sigma_default(prop)
    if paths["eval"].exists():
        ev = json.loads(paths["eval"].read_text())
        sigma = float(ev.get("rmse", sigma))

    df = pd.read_parquet(feat_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Slate features missing required model columns for {prop}: {missing}\n"
            f"This usually means your prop dataset builder didn't create the same feature columns."
        )

    X = df[features].astype(float)
    preds = model.predict(X)

    col = _pred_col(prop)
    df[col] = np.clip(preds, 0, 60)

    # Store sigma used so betslip can compute P(over line)
    df["SIGMA_USED"] = sigma

    out = _slate_preds_path(prop, date_str)
    df.to_parquet(out, index=False)
    print(f"Saved slate preds -> {out}")
    print(f"Rows: {len(df)} | sigma: {sigma:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--prop", choices=["points", "rebounds", "assists", "all"], default="all")
    args = ap.parse_args()

    props = ["points", "rebounds", "assists"] if args.prop == "all" else [args.prop]
    for p in props:
        predict_slate(p, args.date, args.season, args.season_type)


if __name__ == "__main__":
    main()

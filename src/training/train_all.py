"""
train_all.py: train the minutes model and the three prop models on a continuous multi-season history.

Uses the same build_features() the live slate will use. The holdout is chronological: the most
recent season's last 20% of games, by date; everything earlier trains. Writes the filenames the
existing predict code reads, labelled by the latest season.

Known limitation (fixed in milestone 2): MIN_PRED is in-sample for the rows the prop models train
on, so they trust it more than they will at serving time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.seasons import season_for_date
from src.common.validation import validate_training_frame
from src.features.builder import build_features
from src.features.columns import MINUTES_FEATURES, PROP_FEATURES, PROP_TARGETS
from src.features.history import load_history

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"

MINUTES_PARAMS = {"max_depth": 6, "learning_rate": 0.08, "max_iter": 400}
PROP_PARAMS = {"max_depth": 6, "learning_rate": 0.06, "max_iter": 600}
PRED_CLIPS = {"points": (0, 80), "rebounds": (0, 40), "assists": (0, 30)}
MIN_PRED_CLIP = (0, 60)


def holdout_cutoff(df: pd.DataFrame, test_ratio: float = 0.2) -> pd.Timestamp:
    """First date of the holdout: the start of the last `test_ratio` of the LATEST season's rows."""
    seasons = df["GAME_DATE"].map(season_for_date)
    latest = df.loc[seasons == seasons.max(), "GAME_DATE"].sort_values()
    return latest.iloc[int(len(latest) * (1 - test_ratio))]


def _fit(train: pd.DataFrame, features, target: str, params: dict, seed: int):
    validate_training_frame(train, features)  # refuses missing, constant, or mostly empty features
    model = HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **params)
    model.fit(train[features].astype(float), train[target].astype(float))
    return model


def _score(y_true, y_pred, baseline) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "baseline_mae": float(mean_absolute_error(y_true, baseline)),
        "baseline_rmse": float(mean_squared_error(y_true, baseline) ** 0.5),
    }


def _save(artifacts_dir: Path, name: str, label: str, model, features, evaluation: dict) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"{name}_model_{label}.joblib")
    (artifacts_dir / f"{name}_features_{label}.json").write_text(json.dumps(list(features), indent=2))
    (artifacts_dir / f"{name}_eval_{label}.json").write_text(json.dumps(evaluation, indent=2))


def train_models(
    features: pd.DataFrame,
    season_type: str = "Regular Season",
    artifacts_dir: Path = ARTIFACTS_DIR,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, dict]:
    """Train minutes then props on build_features() output. Returns each model's eval object."""
    df = features.loc[~features["IS_UPCOMING"]].dropna(subset=["MIN", "PTS", "REB", "AST"]).copy()
    seasons = sorted(set(df["GAME_DATE"].map(season_for_date)))
    latest = seasons[-1]
    label = f"{latest}_{season_type.replace(' ', '')}"
    cutoff = holdout_cutoff(df, test_ratio)

    def meta(train: pd.DataFrame, test: pd.DataFrame) -> dict:
        return {
            "season": latest,
            "season_type": season_type,
            "seasons": seasons,
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
            "holdout_cutoff": str(cutoff.date()),
            "trained_through": str(train["GAME_DATE"].max().date()),
            "test_date_min": str(test["GAME_DATE"].min().date()),
            "test_date_max": str(test["GAME_DATE"].max().date()),
        }

    results: Dict[str, dict] = {}

    # Stage 1: minutes.
    m = df.dropna(subset=["MIN_ROLL_3", "MIN_ROLL_5", "MIN_ROLL_10"])
    m_train, m_test = m[m["GAME_DATE"] < cutoff], m[m["GAME_DATE"] >= cutoff]
    minutes_model = _fit(m_train, MINUTES_FEATURES, "MIN", MINUTES_PARAMS, seed)
    preds = np.clip(minutes_model.predict(m_test[MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP)
    evaluation = {
        **meta(m_train, m_test),
        "features": MINUTES_FEATURES,
        **_score(m_test["MIN"], preds, m_test["MIN_ROLL_5"]),
    }
    _save(artifacts_dir, "minutes", label, minutes_model, MINUTES_FEATURES, evaluation)
    results["minutes"] = evaluation

    df["MIN_PRED"] = np.clip(minutes_model.predict(df[MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP)

    # Stage 2: props.
    for prop, target in PROP_TARGETS.items():
        feats = PROP_FEATURES[prop]
        needed = [c for c in feats if "ROLL" in c or c == "MIN_PRED"]
        d = df.dropna(subset=needed)
        train, test = d[d["GAME_DATE"] < cutoff], d[d["GAME_DATE"] >= cutoff]
        model = _fit(train, feats, target, PROP_PARAMS, seed)
        preds = np.clip(model.predict(test[feats].astype(float)), *PRED_CLIPS[prop])
        baseline = test["MIN_PRED"] * test[f"{target}_PER_MIN_ROLL_10"]
        evaluation = {**meta(train, test), "features": feats, **_score(test[target], preds, baseline)}
        _save(artifacts_dir, prop, label, model, feats, evaluation)
        results[prop] = evaluation

    return results


def _print_summary(results: Dict[str, dict]) -> None:
    first = next(iter(results.values()))
    print(
        f"\nTrained on {', '.join(first['seasons'])} through {first['trained_through']}; "
        f"holdout {first['test_date_min']} to {first['test_date_max']}"
    )
    print(f"{'model':10s} {'MAE':>7s} {'baseline':>9s} {'lift':>7s}")
    for name, ev in results.items():
        lift = (ev["baseline_mae"] - ev["mae"]) / ev["baseline_mae"]
        print(f"{name:10s} {ev['mae']:7.3f} {ev['baseline_mae']:9.3f} {lift:+7.2%}")


def train_all(
    seasons: Sequence[str],
    season_type: str = "Regular Season",
    use_cache: bool = True,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, dict]:
    players, teams = load_history(seasons, season_type=season_type, use_cache=use_cache)
    results = train_models(build_features(players, teams), season_type=season_type, artifacts_dir=artifacts_dir)
    _print_summary(results)
    return results

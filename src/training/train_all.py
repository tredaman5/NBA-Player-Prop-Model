"""
train_all.py: train the minutes model and the three prop models on a continuous multi-season history.

Uses the same build_features() the live slate will use. The holdout is chronological: the most
recent season's last 20% of games, by date; everything earlier trains. Writes the filenames the
existing predict code reads, labelled by the latest season.

Stage-1 honesty: the prop models take the minutes prediction (MIN_PRED) as a feature. For training
rows it comes from a model that never saw that row (expanding-window blocks of earlier games); for
holdout and live rows it comes from the final model trained on everything before. Otherwise the prop
models would learn from an unrealistically accurate MIN_PRED (4.70 vs 5.31 minutes of error) and
trust it more than they can at serving time.
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
# Chosen up front, not tuned on the holdout: 6 time blocks, honest predictions from the 2nd block on.
OOF_BLOCKS = 6
OOF_MIN_PRIOR_BLOCKS = 1


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


def _block_index(dates: pd.Series, n_blocks: int) -> pd.Series:
    """Split rows into n_blocks consecutive blocks of about equal size by date (a date never straddles two)."""
    inner = [dates.quantile(q) for q in np.linspace(0, 1, n_blocks + 1)[1:-1]]
    index = np.searchsorted(pd.DatetimeIndex(inner).values, dates.values, side="right")
    return pd.Series(index, index=dates.index)


def out_of_fold_minutes(
    train: pd.DataFrame,
    features,
    n_blocks: int = OOF_BLOCKS,
    min_prior_blocks: int = OOF_MIN_PRIOR_BLOCKS,
    seed: int = 42,
) -> pd.Series:
    """
    MIN_PRED for every row of `train`, each from a model fit ONLY on strictly earlier blocks
    (an expanding window), so no row is predicted by a model that has seen it or anything after it.
    The first `min_prior_blocks` blocks have no honest prediction and get NaN.
    """
    block = _block_index(train["GAME_DATE"], n_blocks)
    out = pd.Series(np.nan, index=train.index)
    for k in range(min_prior_blocks, n_blocks):
        fit_rows = train[block < k].dropna(subset=["MIN_ROLL_3", "MIN_ROLL_5", "MIN_ROLL_10"])
        model = _fit(fit_rows, features, "MIN", MINUTES_PARAMS, seed)
        mask = block == k
        out[mask] = np.clip(model.predict(train.loc[mask, features].astype(float)), *MIN_PRED_CLIP)
    return out


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
    oof_blocks: int = OOF_BLOCKS,
    oof_min_prior_blocks: int = OOF_MIN_PRIOR_BLOCKS,
) -> Dict[str, dict]:
    """
    Train minutes then props on build_features() output. Returns each model's eval object.

    oof_blocks=0 switches off out-of-fold minutes (the old in-sample behavior). It exists only so
    the effect can be measured; do not use it for a model you intend to serve.
    """
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

    if oof_blocks:
        in_train = df["GAME_DATE"] < cutoff
        df["MIN_PRED"] = np.nan
        df.loc[in_train, "MIN_PRED"] = out_of_fold_minutes(
            df[in_train], MINUTES_FEATURES, oof_blocks, oof_min_prior_blocks, seed
        )
        df.loc[~in_train, "MIN_PRED"] = np.clip(
            minutes_model.predict(df.loc[~in_train, MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP
        )
    else:
        df["MIN_PRED"] = np.clip(minutes_model.predict(df[MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP)

    scored = df.dropna(subset=["MIN_PRED"])
    before, after = scored[scored["GAME_DATE"] < cutoff], scored[scored["GAME_DATE"] >= cutoff]
    min_pred_report = {
        "oof_blocks": oof_blocks,
        "min_prior_blocks": oof_min_prior_blocks if oof_blocks else None,
        "mae_train": float(mean_absolute_error(before["MIN"], before["MIN_PRED"])),
        "mae_holdout": float(mean_absolute_error(after["MIN"], after["MIN_PRED"])),
        "train_rows_without_min_pred": int(df["MIN_PRED"].isna().sum()),
    }

    # Stage 2: props.
    for prop, target in PROP_TARGETS.items():
        feats = PROP_FEATURES[prop]
        needed = [c for c in feats if "ROLL" in c or c == "MIN_PRED"]
        d = df.dropna(subset=needed)
        train, test = d[d["GAME_DATE"] < cutoff], d[d["GAME_DATE"] >= cutoff]
        model = _fit(train, feats, target, PROP_PARAMS, seed)
        preds = np.clip(model.predict(test[feats].astype(float)), *PRED_CLIPS[prop])
        baseline = test["MIN_PRED"] * test[f"{target}_PER_MIN_ROLL_10"]
        evaluation = {
            **meta(train, test),
            "features": feats,
            "min_pred": min_pred_report,
            **_score(test[target], preds, baseline),
        }
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

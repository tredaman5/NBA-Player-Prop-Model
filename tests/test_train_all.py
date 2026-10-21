import json

import numpy as np
import pandas as pd
import pytest

from src.common.seasons import season_for_date
from src.common.validation import DataQualityError
from src.features.columns import BUILDER_FEATURES
from src.training.train_all import holdout_cutoff, train_models


def _synthetic_features(n_per_season=700, seed=0):
    """Random but well-formed builder output spanning two seasons."""
    rng = np.random.default_rng(seed)
    frames = []
    for start in ("2024-10-22", "2025-10-21"):
        n = n_per_season
        f = pd.DataFrame(
            {
                "GAME_DATE": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 170, n), unit="D"),
                "IS_UPCOMING": False,
            }
        )
        for col in BUILDER_FEATURES:
            f[col] = rng.uniform(0.1, 30.0, n)
        f["IS_HOME"] = rng.integers(0, 2, n)
        f["IS_B2B"] = rng.integers(0, 2, n)
        f["DAYS_REST"] = rng.integers(1, 6, n)
        f["MIN"] = np.clip(0.8 * f["MIN_ROLL_5"] + rng.normal(0, 4, n), 0, 48)
        f["PTS"] = np.clip(f["MIN"] * 0.5 + rng.normal(0, 3, n), 0, None)
        f["REB"] = np.clip(f["MIN"] * 0.2 + rng.normal(0, 1.5, n), 0, None)
        f["AST"] = np.clip(f["MIN"] * 0.12 + rng.normal(0, 1.2, n), 0, None)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_holdout_is_the_last_fifth_of_the_latest_season():
    df = _synthetic_features()
    cutoff = holdout_cutoff(df, 0.2)
    assert season_for_date(cutoff) == "2025-26"
    latest = df[df["GAME_DATE"].map(season_for_date) == "2025-26"]
    share = (latest["GAME_DATE"] >= cutoff).mean()
    assert 0.15 < share < 0.25


def test_train_models_writes_artifacts_and_tests_only_on_later_dates(tmp_path):
    results = train_models(_synthetic_features(), artifacts_dir=tmp_path)

    assert set(results) == {"minutes", "points", "rebounds", "assists"}
    for name in results:
        for kind, ext in (("model", "joblib"), ("features", "json"), ("eval", "json")):
            assert (tmp_path / f"{name}_{kind}_2025-26_RegularSeason.{ext}").exists()

    ev = json.loads((tmp_path / "points_eval_2025-26_RegularSeason.json").read_text())
    assert ev["seasons"] == ["2024-25", "2025-26"]
    assert pd.Timestamp(ev["trained_through"]) < pd.Timestamp(ev["holdout_cutoff"]) <= pd.Timestamp(ev["test_date_min"])
    assert ev["rows_train"] > ev["rows_test"] > 0
    assert np.isfinite(ev["mae"]) and np.isfinite(ev["baseline_mae"])
    assert len(ev["features"]) == 26


def test_upcoming_rows_are_never_trained_on(tmp_path):
    base = _synthetic_features()
    upcoming = base.iloc[:60].copy()
    upcoming["IS_UPCOMING"] = True
    upcoming[["MIN", "PTS", "REB", "AST"]] = np.nan
    without = train_models(base, artifacts_dir=tmp_path / "a")
    with_upcoming = train_models(pd.concat([base, upcoming], ignore_index=True), artifacts_dir=tmp_path / "b")
    assert without["points"]["rows_train"] == with_upcoming["points"]["rows_train"]
    assert without["points"]["rows_test"] == with_upcoming["points"]["rows_test"]


def test_a_constant_feature_stops_training(tmp_path):
    df = _synthetic_features()
    df["DAYS_REST"] = 3  # what a source that silently went empty looks like
    with pytest.raises(DataQualityError, match="constant"):
        train_models(df, artifacts_dir=tmp_path)

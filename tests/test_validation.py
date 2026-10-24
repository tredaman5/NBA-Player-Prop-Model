"""
Tests for src/common/validation.py: data-quality gates that stop the pipeline when a
source degrades instead of letting it quietly train on junk.

The motivating failure: a per-game pull returned 0 rows for a whole season, a third of
its calls failed, and the build trained on the empty feature anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.validation import (
    DataQualityError,
    validate_player_gamelog,
    validate_team_gamelog,
    validate_training_frame,
)


def _player_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g1", "g2"],
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]),
            "PLAYER_ID": [1, 2, 1],
            "PLAYER_NAME": ["A", "B", "A"],
            "TEAM_ID": [10, 20, 10],
            "TEAM_ABBREVIATION": ["AAA", "BBB", "AAA"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA", "AAA @ CCC"],
            "MIN": [30, 25, 0],  # a 0-minute row is a normal DNP
            "PTS": [20, 10, 0],
            "REB": [5, 3, 0],
            "AST": [4, 2, 0],
            "FGA": [15, 8, 0],
            "FTA": [4, 2, 0],
            "FG3A": [5, 3, 0],
            "TOV": [2, 1, 0],
        }
    )


def _team_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g1"],
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "TEAM_ID": [10, 20],
            "TEAM_ABBREVIATION": ["AAA", "BBB"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "PTS": [110, 100],
            "REB": [45, 40],
            "AST": [25, 22],
            "FGA": [88, 85],
            "FTA": [20, 18],
            "TOV": [12, 14],
        }
    )


def test_clean_player_log_passes():
    validate_player_gamelog(_player_log())


def test_empty_player_log_is_rejected():
    with pytest.raises(DataQualityError, match="empty"):
        validate_player_gamelog(_player_log().iloc[0:0])


def test_missing_column_points_at_a_schema_change():
    with pytest.raises(DataQualityError, match="schema"):
        validate_player_gamelog(_player_log().drop(columns=["TOV"]))


def test_duplicate_player_game_rows_are_rejected():
    df = pd.concat([_player_log(), _player_log().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataQualityError, match="duplicate"):
        validate_player_gamelog(df)


def test_missing_ids_are_rejected():
    df = _player_log()
    df.loc[1, "PLAYER_ID"] = np.nan
    with pytest.raises(DataQualityError, match="PLAYER_ID"):
        validate_player_gamelog(df)


def test_impossible_values_are_rejected():
    df = _player_log()
    df.loc[0, "MIN"] = 190  # nobody plays 190 minutes
    df.loc[1, "PTS"] = -3
    with pytest.raises(DataQualityError) as exc:
        validate_player_gamelog(df)
    assert "MIN" in str(exc.value) and "PTS" in str(exc.value)  # reports every problem at once


def test_clean_team_log_passes():
    validate_team_gamelog(_team_log())


def test_team_log_with_a_missing_side_is_rejected():
    with pytest.raises(DataQualityError, match="exactly 2"):
        validate_team_gamelog(_team_log().iloc[[0]])


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=200), "b": rng.integers(0, 2, size=200)})


def test_healthy_training_frame_passes():
    validate_training_frame(_frame(), ["a", "b"])


def test_constant_feature_is_rejected():
    df = _frame()
    df["b"] = 0  # a source that silently went empty looks like this
    with pytest.raises(DataQualityError, match="constant"):
        validate_training_frame(df, ["a", "b"])


def test_mostly_missing_feature_is_rejected():
    df = _frame()
    df.loc[df.index[:150], "a"] = np.nan
    with pytest.raises(DataQualityError, match="missing"):
        validate_training_frame(df, ["a", "b"])


def test_feature_absent_from_the_dataset_is_rejected():
    with pytest.raises(DataQualityError, match="not in the dataset"):
        validate_training_frame(_frame(), ["a", "ghost"])


def test_an_empty_log_raises_the_empty_error_which_is_still_a_data_quality_error():
    from src.common.validation import EmptyDataError

    with pytest.raises(EmptyDataError):
        validate_player_gamelog(_player_log().iloc[0:0])
    with pytest.raises(EmptyDataError):
        validate_team_gamelog(_team_log().iloc[0:0])
    assert issubclass(EmptyDataError, DataQualityError)


def test_serving_check_accepts_a_constant_column_because_a_small_slate_can_have_one():
    from src.common.validation import validate_serving_frame

    validate_serving_frame(pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 4.0]}), ["a", "b"])


def test_serving_check_rejects_a_missing_or_mostly_null_feature():
    from src.common.validation import validate_serving_frame

    with pytest.raises(DataQualityError, match="not in the dataset"):
        validate_serving_frame(pd.DataFrame({"a": [1.0]}), ["a", "ghost"])
    with pytest.raises(DataQualityError, match="missing"):
        validate_serving_frame(pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan]}), ["a"])

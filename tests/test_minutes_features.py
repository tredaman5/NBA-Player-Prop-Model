"""
Regression test for the minutes-model leakage bug: PTS/REB/AST/FGA/FTA/FG3A/TOV
are this game's actual box score and must never be selected as features for
predicting this game's minutes.
"""

from __future__ import annotations

import pandas as pd

from src.minutes.train import _select_features


BANNED_COLUMNS = {"PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"}


def test_select_features_excludes_current_game_boxscore():
    # Simulate a dataset that has both the legitimate pregame features and
    # the banned current-game boxscore columns (as build_dataset.py produces).
    df = pd.DataFrame(
        {
            "IS_HOME": [1, 0],
            "DAYS_REST": [2, 1],
            "IS_B2B": [0, 1],
            "MIN_ROLL_3": [30.0, 28.0],
            "MIN_ROLL_5": [29.0, 27.0],
            "MIN_ROLL_10": [28.5, 26.5],
            "PTS": [20, 15],
            "REB": [5, 8],
            "AST": [6, 3],
            "FGA": [15, 12],
            "FTA": [4, 2],
            "FG3A": [3, 1],
            "TOV": [2, 3],
        }
    )

    features = _select_features(df)

    leaked = BANNED_COLUMNS & set(features)
    assert not leaked, f"Leaky current-game columns selected as features: {leaked}"
    assert set(features) == {
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
    }

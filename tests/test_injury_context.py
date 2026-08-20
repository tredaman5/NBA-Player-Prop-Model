"""
Tests for src/common/injury_context.py's teammates-out feature:
- uses the PRIOR game's rotation, never the current game's box score
  (that would be leakage/circular -- an inactive player has no MIN row
  in the current game anyway)
- correctly counts overlap between prior rotation and tonight's inactive list
- first game of the season (no prior game) doesn't crash, just reports 0
"""

from __future__ import annotations

import pandas as pd

from src.common.injury_context import build_teammates_out_feature


def _player_minutes_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Game 1: P1, P2 are rotation (>=15 min); P3 is not (5 min).
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "MIN": 30},
            {"PLAYER_ID": 2, "TEAM_ABBREVIATION": "A", "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "MIN": 20},
            {"PLAYER_ID": 3, "TEAM_ABBREVIATION": "A", "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "MIN": 5},
            # Game 2: P2 is out (no row at all); P4 fills in as rotation.
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g2", "GAME_DATE": "2024-01-03", "MIN": 28},
            {"PLAYER_ID": 4, "TEAM_ABBREVIATION": "A", "GAME_ID": "g2", "GAME_DATE": "2024-01-03", "MIN": 18},
            # Game 3: P2 is back.
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g3", "GAME_DATE": "2024-01-05", "MIN": 25},
            {"PLAYER_ID": 2, "TEAM_ABBREVIATION": "A", "GAME_ID": "g3", "GAME_DATE": "2024-01-05", "MIN": 22},
        ]
    )


def _inactive_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"GAME_ID": "g2", "TEAM_ABBREVIATION": "A", "PLAYER_ID": 2, "PLAYER_NAME": "P2"},
        ]
    )


def test_first_game_has_no_prior_rotation_and_does_not_crash():
    out = build_teammates_out_feature(_player_minutes_df(), _inactive_df(), min_threshold=15.0)
    g1 = out[out["GAME_ID"] == "g1"].iloc[0]
    assert g1["TEAMMATES_OUT_ROTATION"] == 0


def test_counts_overlap_between_prior_rotation_and_tonights_inactive_list():
    out = build_teammates_out_feature(_player_minutes_df(), _inactive_df(), min_threshold=15.0)

    # g2: prior rotation (from g1) = {P1, P2}; inactive tonight = {P2} -> overlap 1.
    g2 = out[out["GAME_ID"] == "g2"].iloc[0]
    assert g2["TEAMMATES_OUT_ROTATION"] == 1

    # g3: prior rotation (from g2) = {P1, P4}; nobody inactive tonight -> overlap 0.
    g3 = out[out["GAME_ID"] == "g3"].iloc[0]
    assert g3["TEAMMATES_OUT_ROTATION"] == 0


def test_does_not_use_current_game_box_score():
    # P2 has no row in g2 (they were out) -- if the function accidentally
    # used g2's own rotation instead of g1's prior rotation, P2 wouldn't be
    # in the rotation set at all (no row to be >= threshold) and the
    # overlap would come out 0 instead of 1.
    out = build_teammates_out_feature(_player_minutes_df(), _inactive_df(), min_threshold=15.0)
    g2 = out[out["GAME_ID"] == "g2"].iloc[0]
    assert g2["TEAMMATES_OUT_ROTATION"] == 1

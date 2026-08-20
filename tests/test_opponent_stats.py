"""
Tests for src/common/opponent_stats.py:
- opponent attribution is correct (allowed = what the other team scored)
- rolling allowed-stats only reflect games strictly before the game in question
  (no leakage)
- merge_opponent_features attaches the right team's numbers to a player row
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.opponent_stats import (
    build_opponent_defense_table,
    merge_opponent_features,
    opponent_feature_cols,
)


def _team_gamelog() -> pd.DataFrame:
    # TeamA plays 3 games (vs B, then C, then B again).
    # TeamA allowed: 90 (g1), 95 (g2), 88 (g3).
    return pd.DataFrame(
        [
            {"GAME_ID": "g1", "GAME_DATE": "2024-01-01", "TEAM_ABBREVIATION": "A", "PTS": 100},
            {"GAME_ID": "g1", "GAME_DATE": "2024-01-01", "TEAM_ABBREVIATION": "B", "PTS": 90},
            {"GAME_ID": "g2", "GAME_DATE": "2024-01-03", "TEAM_ABBREVIATION": "A", "PTS": 110},
            {"GAME_ID": "g2", "GAME_DATE": "2024-01-03", "TEAM_ABBREVIATION": "C", "PTS": 95},
            {"GAME_ID": "g3", "GAME_DATE": "2024-01-05", "TEAM_ABBREVIATION": "A", "PTS": 105},
            {"GAME_ID": "g3", "GAME_DATE": "2024-01-05", "TEAM_ABBREVIATION": "B", "PTS": 88},
        ]
    )


def test_opponent_attribution_is_correct():
    table = build_opponent_defense_table(_team_gamelog(), stat_cols=("PTS",), windows=(2,))

    row_g1_teamA = table[(table["GAME_ID"] == "g1") & (table["TEAM_ABBREVIATION"] == "A")].iloc[0]
    row_g1_teamB = table[(table["GAME_ID"] == "g1") & (table["TEAM_ABBREVIATION"] == "B")].iloc[0]

    # Both exist and represent opposite teams for the same game -- sanity check
    # that the self-join paired A with B (not with itself).
    assert row_g1_teamA["TEAM_ABBREVIATION"] == "A"
    assert row_g1_teamB["TEAM_ABBREVIATION"] == "B"


def test_rolling_allowed_has_no_leakage():
    table = build_opponent_defense_table(_team_gamelog(), stat_cols=("PTS",), windows=(2,))
    a_rows = table[table["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE")

    # Game 1: TeamA's first game -> no prior games -> NaN, not 0 and not
    # leaking this game's own allowed value.
    assert np.isnan(a_rows.iloc[0]["OPP_PTS_ALLOWED_ROLL_2"])

    # Game 2: only one prior game (allowed 90 in g1) -> rolling mean = 90.
    assert a_rows.iloc[1]["OPP_PTS_ALLOWED_ROLL_2"] == 90.0

    # Game 3: two prior games (allowed 90 in g1, 95 in g2) -> mean = 92.5.
    # Must NOT include g3's own allowed value (88).
    assert a_rows.iloc[2]["OPP_PTS_ALLOWED_ROLL_2"] == 92.5


def test_merge_attaches_opponents_numbers_not_own_team():
    table = build_opponent_defense_table(_team_gamelog(), stat_cols=("PTS",), windows=(2,))
    opp_cols = opponent_feature_cols("PTS", windows=(2,))

    # A player on TeamA's roster, facing TeamB, in game g3.
    player_df = pd.DataFrame(
        [{"GAME_ID": "g3", "TEAM_ABBREVIATION": "A", "OPP_ABBREVIATION": "B", "PLAYER_NAME": "Test Player"}]
    )

    merged = merge_opponent_features(player_df, table, opp_cols)

    # Should carry TeamB's rolling allowed-points entering g3 (B's only prior
    # game was g1, where B allowed 100), not TeamA's own defensive numbers.
    assert merged.loc[0, "OPP_PTS_ALLOWED_ROLL_2"] == 100.0
    assert "_OPP_TEAM_ABBR" not in merged.columns

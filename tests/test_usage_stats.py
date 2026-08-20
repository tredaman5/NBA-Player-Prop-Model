"""
Tests for src/common/usage_stats.py:
- usage rate is computed correctly (player's share of team's possessions-used)
- rolling values have no leakage (never include the game they're attached to)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.usage_stats import add_usage_rate_rolling


def _player_df() -> pd.DataFrame:
    # Player P1 on team A, 3 games. Team A's totals are per-game, distinct
    # from any other player's row.
    return pd.DataFrame(
        [
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "FGA": 10, "FTA": 5, "TOV": 2},
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g2", "GAME_DATE": "2024-01-03", "FGA": 20, "FTA": 0, "TOV": 0},
            {"PLAYER_ID": 1, "TEAM_ABBREVIATION": "A", "GAME_ID": "g3", "GAME_DATE": "2024-01-05", "FGA": 5, "FTA": 5, "TOV": 5},
        ]
    )


def _team_gamelog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"GAME_ID": "g1", "TEAM_ABBREVIATION": "A", "FGA": 100, "FTA": 20, "TOV": 10},  # team possessions = 100+8.8+10=118.8
            {"GAME_ID": "g2", "TEAM_ABBREVIATION": "A", "FGA": 100, "FTA": 0, "TOV": 0},     # team possessions = 100
            {"GAME_ID": "g3", "TEAM_ABBREVIATION": "A", "FGA": 50, "FTA": 0, "TOV": 0},      # team possessions = 50
        ]
    )


def test_first_game_has_no_prior_usage_rate():
    out = add_usage_rate_rolling(_player_df(), _team_gamelog(), windows=(2,))
    g1 = out[out["GAME_ID"] == "g1"].iloc[0]
    assert np.isnan(g1["USAGE_RATE_ROLL_2"])


def test_rolling_usage_rate_uses_only_prior_games():
    out = add_usage_rate_rolling(_player_df(), _team_gamelog(), windows=(2,))

    # g1 usage: (10 + 0.44*5 + 2) / 118.8 = 14.2/118.8 = 0.11953...
    # g2 usage: (20 + 0 + 0) / 100 = 0.20
    # g3's prior-2 rolling average should be mean(g1_usage, g2_usage), NOT
    # including g3's own usage (5 + 0.44*5 + 5)/50 = 12.2/50 = 0.244.
    g1_usage = (10 + 0.44 * 5 + 2) / 118.8
    g2_usage = (20 + 0.44 * 0 + 0) / 100
    expected_g3_rolling = (g1_usage + g2_usage) / 2

    g3 = out[out["GAME_ID"] == "g3"].iloc[0]
    assert abs(g3["USAGE_RATE_ROLL_2"] - expected_g3_rolling) < 1e-9

    # Sanity: this must differ from what you'd get by (wrongly) including g3.
    g3_usage = (5 + 0.44 * 5 + 5) / 50
    wrong_value_if_leaked = (g2_usage + g3_usage) / 2
    assert abs(g3["USAGE_RATE_ROLL_2"] - wrong_value_if_leaked) > 1e-6

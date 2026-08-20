"""
usage_stats.py

Rolling usage-rate feature: what share of the team's total possessions-used
(FGA + 0.44*FTA + TOV) a player personally accounted for. Distinct from the
existing PTS_PER_MIN_ROLL/FGA_PER_MIN_ROLL features because it normalizes
for team pace -- a player on a fast team racks up more attempts per minute
than an equally-featured player on a slow team, which those per-minute
features don't separate out.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_usage_rate_rolling(
    player_df: pd.DataFrame,
    team_gamelog: pd.DataFrame,
    windows: Iterable[int] = (3, 5, 10),
) -> pd.DataFrame:
    """
    Adds USAGE_RATE_ROLL_{w} columns to player_df.

    Computes each historical game's actual usage rate (player's FGA+0.44*FTA+TOV
    divided by the team's same-game total), then rolls it with shift(1) so a
    game's value only reflects games strictly before it -- no leakage, even
    though the raw per-game usage rate itself uses that game's real numbers.

    player_df must have GAME_ID, TEAM_ABBREVIATION, PLAYER_ID, GAME_DATE,
    FGA, FTA, TOV. team_gamelog must have GAME_ID, TEAM_ABBREVIATION, FGA,
    FTA, TOV (team totals for that game) -- the same team-level log already
    used for opponent_stats.py, no extra fetch needed.
    """
    windows = list(windows)

    team_totals = team_gamelog[["GAME_ID", "TEAM_ABBREVIATION", "FGA", "FTA", "TOV"]].rename(
        columns={"FGA": "TEAM_FGA", "FTA": "TEAM_FTA", "TOV": "TEAM_TOV"}
    )

    out = player_df.merge(team_totals, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    player_possessions = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]
    team_possessions = out["TEAM_FGA"] + 0.44 * out["TEAM_FTA"] + out["TEAM_TOV"]
    team_possessions_safe = team_possessions.replace(0, np.nan)

    out["_USAGE_RATE"] = (player_possessions / team_possessions_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.sort_values(["PLAYER_ID", "GAME_DATE"])
    for w in windows:
        out[f"USAGE_RATE_ROLL_{w}"] = (
            out.groupby("PLAYER_ID")["_USAGE_RATE"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    return out.drop(columns=["TEAM_FGA", "TEAM_FTA", "TEAM_TOV", "_USAGE_RATE"])


def usage_rate_feature_cols(windows: Iterable[int] = (3, 5, 10)) -> list[str]:
    return [f"USAGE_RATE_ROLL_{w}" for w in windows]

"""
builder.py: the one feature builder.

Training, serving and the backtest all call build_features(), so a feature can never exist
in one and be stale in another. Pure functions over DataFrames: no file or network I/O.

Every rolling feature is a mean over the games BEFORE the one it describes (shift(1)), so
a game's features never include its own box score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.opponent_stats import (
    build_opponent_defense_table,
    merge_opponent_features,
    opponent_feature_cols,
)
from src.common.usage_stats import add_usage_rate_rolling
from src.features.columns import (
    MIN_WINDOWS,
    OPP_STATS,
    OPP_WINDOWS,
    RATE_STATS,
    TEAM_PTS_WINDOWS,
    USAGE_WINDOWS,
)

PLAYER_COLS = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "MATCHUP", "MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV",
]
TEAM_COLS = [
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
    "PTS", "REB", "AST", "FGA", "FTA", "TOV",
]


def _rolling_prior_mean(df: pd.DataFrame, by: str, column: str, window: int) -> pd.Series:
    """Mean of `column` over the previous `window` rows of each `by` group (current row excluded).
    df must already be sorted by (by, date)."""
    return df.groupby(by)[column].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def _add_context(p: pd.DataFrame) -> pd.DataFrame:
    """IS_HOME and OPP_ABBREVIATION from MATCHUP ("LAL vs. GSW" is home, "LAL @ GSW" is away)."""
    matchup = p["MATCHUP"].astype(str)
    p["IS_HOME"] = matchup.str.contains("vs.", regex=False).astype(int)
    p["OPP_ABBREVIATION"] = matchup.str.split().str[-1]
    return p


def _add_rest(p: pd.DataFrame) -> pd.DataFrame:
    """Days since the player's previous game (3 when there is none), capped at 14."""
    prev = p.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    p["DAYS_REST"] = (p["GAME_DATE"] - prev).dt.days.fillna(3).clip(lower=0, upper=14)
    p["IS_B2B"] = (p["DAYS_REST"] <= 1).astype(int)
    return p


def _add_rolling_minutes(p: pd.DataFrame) -> pd.DataFrame:
    p["MIN"] = pd.to_numeric(p["MIN"], errors="coerce")
    for w in MIN_WINDOWS:
        p[f"MIN_ROLL_{w}"] = _rolling_prior_mean(p, "PLAYER_ID", "MIN", w)
    return p


def _add_per_minute_rates(p: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-minute production. A 0-minute game counts as a 0 rate."""
    minutes = pd.to_numeric(p["MIN"], errors="coerce").replace(0, np.nan)
    for stat in RATE_STATS:
        rate = (pd.to_numeric(p[stat], errors="coerce") / minutes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p[f"_{stat}_RATE"] = rate
        for w in MIN_WINDOWS:
            p[f"{stat}_PER_MIN_ROLL_{w}"] = _rolling_prior_mean(p, "PLAYER_ID", f"_{stat}_RATE", w)
    return p.drop(columns=[f"_{s}_RATE" for s in RATE_STATS])


def _add_usage(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling share of the team's possessions the player used (FGA + 0.44*FTA + TOV)."""
    return add_usage_rate_rolling(p, t, windows=USAGE_WINDOWS).reset_index(drop=True)


def _add_team_points(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of the player's own team's points in its previous games."""
    team = t[["GAME_ID", "TEAM_ID", "GAME_DATE", "PTS"]].copy()
    team["PTS"] = pd.to_numeric(team["PTS"], errors="coerce")
    team = team.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    cols = []
    for w in TEAM_PTS_WINDOWS:
        col = f"TEAM_PTS_ROLL_{w}"
        team[col] = _rolling_prior_mean(team, "TEAM_ID", "PTS", w)
        cols.append(col)
    return p.merge(team[["TEAM_ID", "GAME_ID"] + cols], on=["TEAM_ID", "GAME_ID"], how="left")


def _add_opponent(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling points/rebounds/assists the player's opponent has allowed in its previous games."""
    defense = build_opponent_defense_table(t, stat_cols=OPP_STATS, windows=OPP_WINDOWS)
    cols = [c for stat in OPP_STATS for c in opponent_feature_cols(stat, OPP_WINDOWS)]
    return merge_opponent_features(p, defense, cols)


def build_features(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per input player-game with its features, sorted by (GAME_DATE, GAME_ID, PLAYER_ID)."""
    p = player_games[PLAYER_COLS].copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])
    t = team_games[TEAM_COLS].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])
    n_rows = len(p)

    p = _add_context(p)
    p = p.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    p = _add_rest(p)
    p = _add_rolling_minutes(p)
    p = _add_per_minute_rates(p)
    p = _add_usage(p, t)
    p = _add_team_points(p, t)
    p = _add_opponent(p, t)

    if len(p) != n_rows:
        raise ValueError(
            f"feature merges changed the row count ({n_rows} -> {len(p)}); check for duplicate team or player rows"
        )
    return p.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)

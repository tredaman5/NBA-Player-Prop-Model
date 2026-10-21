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

from src.features.columns import MIN_WINDOWS, RATE_STATS

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


def build_features(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per input player-game with its features, sorted by (GAME_DATE, GAME_ID, PLAYER_ID)."""
    p = player_games[PLAYER_COLS].copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])

    p = _add_context(p)
    p = p.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    p = _add_rest(p)
    p = _add_rolling_minutes(p)
    p = _add_per_minute_rates(p)

    return p.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)

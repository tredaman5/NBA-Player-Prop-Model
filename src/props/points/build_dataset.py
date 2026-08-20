"""
build_dataset.py (points props)

Builds a player-game dataset for training a POINTS model.
Uses:
- actual MIN (target-side training)
- MIN_PRED (feature for what you'd have known pregame, once you deploy)

Output:
- data/processed/points_dataset_<season>_<season_type>.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.opponent_stats import (
    build_opponent_defense_table,
    fetch_league_team_gamelogs,
    merge_opponent_features,
    opponent_feature_cols,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def _default_minutes_pred_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"minutes_with_pred_{season}_{season_type.replace(' ', '')}.parquet"


def _add_rolling_opportunity(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    """
    Build shifted rolling opportunity features so no leakage:
    - FGA per minute rolling
    - FTA per minute rolling
    - 3PA per minute rolling
    - PTS per minute rolling (can be useful but be careful not to overfit)
    """
    out = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    # Protect against divide-by-zero
    min_safe = out["MIN"].replace(0, np.nan)

    out["FGA_PER_MIN"] = out["FGA"] / min_safe
    out["FTA_PER_MIN"] = out["FTA"] / min_safe
    out["FG3A_PER_MIN"] = out["FG3A"] / min_safe
    out["PTS_PER_MIN"] = out["PTS"] / min_safe

    # Replace infinities/nans created by 0 minutes
    for c in ["FGA_PER_MIN", "FTA_PER_MIN", "FG3A_PER_MIN", "PTS_PER_MIN"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in windows:
        for base in ["FGA_PER_MIN", "FTA_PER_MIN", "FG3A_PER_MIN", "PTS_PER_MIN"]:
            out[f"{base}_ROLL_{w}"] = (
                out.groupby("PLAYER_ID")[base]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
    return out


def _add_rolling_team_context(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    Simple team context from player logs:
    - TEAM_PTS_ROLL_w: average team points in last w games
      (using summed player PTS per TEAM_ID per GAME_ID)
    """
    out = df.copy()

    # Team points per game
    team_game_pts = (
        out.groupby(["TEAM_ID", "GAME_ID", "GAME_DATE"], as_index=False)["PTS"]
        .sum()
        .rename(columns={"PTS": "TEAM_PTS"})
    )

    team_game_pts = team_game_pts.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    for w in windows:
        team_game_pts[f"TEAM_PTS_ROLL_{w}"] = (
            team_game_pts.groupby("TEAM_ID")["TEAM_PTS"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    out = out.merge(
        team_game_pts[["TEAM_ID", "GAME_ID"] + [f"TEAM_PTS_ROLL_{w}" for w in windows]],
        on=["TEAM_ID", "GAME_ID"],
        how="left",
    )

    return out


def build_points_dataset(
    season: str,
    season_type: str = "Regular Season",
    minutes_with_pred_path: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Builds and saves points training dataset.

    Target: PTS
    Features include:
    - MIN_PRED, rolling MIN, rest, home/away
    - rolling opportunity rates (FGA/FTA/3PA per min)
    - light team context
    """
    if minutes_with_pred_path is None:
        minutes_with_pred_path = _default_minutes_pred_path(season, season_type)

    if not minutes_with_pred_path.exists():
        raise FileNotFoundError(
            f"Minutes-with-pred file not found: {minutes_with_pred_path}\n"
            f"Run: python -m src.minutes.predict"
        )

    df = pd.read_parquet(minutes_with_pred_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Keep only rows with real boxscore stats
    needed = ["PTS", "MIN", "FGA", "FTA", "FG3A", "MIN_PRED"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in minutes-with-pred dataset.")

    # Basic cleanup
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.dropna(subset=["PTS", "MIN"]).copy()

    # Rolling opportunity + team context
    df = _add_rolling_opportunity(df)
    df = _add_rolling_team_context(df)

    # Opponent defense context (how many points this opponent has been
    # allowing recently, shifted so no future info leaks in)
    team_gamelog = fetch_league_team_gamelogs(season, season_type=season_type, use_cache=use_cache)
    defense_table = build_opponent_defense_table(team_gamelog, stat_cols=("PTS",))
    opp_cols = opponent_feature_cols("PTS")
    df = merge_opponent_features(df, defense_table, opp_cols)

    # Final feature set (keep it readable)
    feature_cols = [
        # from minutes dataset
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        "MIN_PRED",
        # opportunity rates
        "FGA_PER_MIN_ROLL_3",
        "FGA_PER_MIN_ROLL_5",
        "FGA_PER_MIN_ROLL_10",
        "FTA_PER_MIN_ROLL_3",
        "FTA_PER_MIN_ROLL_5",
        "FTA_PER_MIN_ROLL_10",
        "FG3A_PER_MIN_ROLL_3",
        "FG3A_PER_MIN_ROLL_5",
        "FG3A_PER_MIN_ROLL_10",
        "PTS_PER_MIN_ROLL_3",
        "PTS_PER_MIN_ROLL_5",
        "PTS_PER_MIN_ROLL_10",
        # team context
        "TEAM_PTS_ROLL_5",
        "TEAM_PTS_ROLL_10",
        # opponent defense context
        *opp_cols,
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPP_ABBREVIATION",
        "PTS",  # target
        "MIN",  # actual minutes (for analysis)
    ] + feature_cols

    out = df[keep_cols].copy()

    out_path = DATA_PROCESSED / f"points_dataset_{season}_{season_type.replace(' ', '')}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved points dataset -> {out_path}")
    print(f"Rows: {len(out):,} | Features: {len(feature_cols)}")

    return out


if __name__ == "__main__":
    build_points_dataset(season="2025-26", season_type="Regular Season")

"""
build_dataset.py (minutes)

Builds a player-game dataset for training a MINUTES model.
Designed to work for completed seasons and in-progress seasons.

Data source: nba_api.stats.endpoints.leaguegamelog (player-level logs).
Output: data/processed/minutes_dataset_<season>_<season_type>.parquet
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.common.injury_context import build_teammates_out_feature, fetch_game_inactive_players

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _season_str(season_start_year: int) -> str:
    yy = str((season_start_year + 1) % 100).zfill(2)
    return f"{season_start_year}-{yy}"


def fetch_league_player_gamelogs(
    season: str,
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_sec: float = 0.8,
) -> pd.DataFrame:
    """
    Pull all PLAYER game logs for a season (league-wide) in one shot.

    Returns columns including:
    GAME_ID, GAME_DATE, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION,
    MATCHUP, MIN, PTS, REB, AST, etc.
    """
    _ensure_dir(DATA_RAW)
    cache_path = DATA_RAW / f"league_player_gamelog_{season}_{season_type.replace(' ', '')}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    time.sleep(sleep_sec)
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",  # P = Player logs
    )
    df = resp.get_data_frames()[0].copy()

    # Normalize date
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df.to_parquet(cache_path, index=False)
    return df


def _add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - IS_HOME (from MATCHUP "vs." or "@")
    - OPP_ABBREVIATION (parsed from MATCHUP)
    """
    out = df.copy()

    # MATCHUP looks like "LAL vs. GSW" or "LAL @ GSW"
    matchup = out["MATCHUP"].astype(str)
    out["IS_HOME"] = matchup.str.contains("vs.", regex=False).astype(int)

    # Opp is the last token
    out["OPP_ABBREVIATION"] = matchup.str.split().str[-1]
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - DAYS_REST: days since previous game for that player
    - IS_B2B: back-to-back indicator (<= 1 day rest)
    """
    out = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    prev_date = out.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    out["DAYS_REST"] = (out["GAME_DATE"] - prev_date).dt.days

    # For first game, rest is unknown; set to median-ish value (3) or 0
    out["DAYS_REST"] = out["DAYS_REST"].fillna(3).clip(lower=0, upper=14)
    out["IS_B2B"] = (out["DAYS_REST"] <= 1).astype(int)

    return out


def _add_rolling_minutes(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    """
    Adds shifted rolling averages of MIN so no leakage:
    MIN_ROLL_3 = mean of previous 3 games' minutes (not including current)
    """
    out = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")

    for w in windows:
        out[f"MIN_ROLL_{w}"] = (
            out.groupby("PLAYER_ID")["MIN"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return out


def build_minutes_dataset(
    season_start_year: int,
    season_type: str = "Regular Season",
    cutoff_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build and save the minutes dataset.

    cutoff_date: optional ISO date string "YYYY-MM-DD".
                 Useful for in-progress seasons (build up to that date).
    """
    _ensure_dir(DATA_PROCESSED)

    season = _season_str(season_start_year)
    df = fetch_league_player_gamelogs(season, season_type=season_type, use_cache=use_cache)

    # Keep only valid player-games with minutes
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.dropna(subset=["MIN"]).copy()

    # If you want only up to a certain date (in-progress season)
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        df = df[df["GAME_DATE"] <= cutoff].copy()

    df = _add_context_columns(df)
    df = _add_time_features(df)
    df = _add_rolling_minutes(df)

    # Rotation-teammates-out feature: how many of this team's rotation
    # players (from their prior game) are on tonight's official inactive
    # list. Pregame-known information, not leakage -- see injury_context.py.
    game_ids = df["GAME_ID"].unique().tolist()
    inactive_df = fetch_game_inactive_players(game_ids, season=season, season_type=season_type, use_cache=use_cache)
    teammates_out = build_teammates_out_feature(df, inactive_df)
    df = df.merge(
        teammates_out[["TEAM_ABBREVIATION", "GAME_ID", "TEAMMATES_OUT_ROTATION"]],
        on=["TEAM_ABBREVIATION", "GAME_ID"],
        how="left",
    )
    df["TEAMMATES_OUT_ROTATION"] = df["TEAMMATES_OUT_ROTATION"].fillna(0)

    # Select a clean set of columns (you can add more later)
    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPP_ABBREVIATION",
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        "TEAMMATES_OUT_ROTATION",
        # Optional: keep box-score stats for later feature ideas
        "PTS",
        "REB",
        "AST",
        "FGA",
        "FTA",
        "FG3A",
        "TOV",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    # Save
    out_path = DATA_PROCESSED / f"minutes_dataset_{season}_{season_type.replace(' ', '')}.parquet"
    out.to_parquet(out_path, index=False)

    return out

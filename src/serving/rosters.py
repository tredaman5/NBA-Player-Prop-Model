"""
rosters.py: who is on which team right now.

A player's team for tonight comes from the current roster, not from their last game, because about
a quarter of players change teams between seasons.
"""

from __future__ import annotations

import pandas as pd

from src.common.validation import DataQualityError

ROSTER_COLUMNS = ["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_ABBREVIATION"]
MIN_TEAMS = 25


def validate_roster(df: pd.DataFrame) -> None:
    missing = [c for c in ROSTER_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"roster is missing columns {missing}; the provider's schema may have changed")
    if df.empty:
        raise DataQualityError("roster is empty")
    problems = []
    teams = df["TEAM_ID"].nunique()
    if teams < MIN_TEAMS:
        problems.append(f"only {teams} teams on the roster list (expected about 30); the pull looks partial")
    dupes = int(df["PERSON_ID"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicate PERSON_ID rows")
    if problems:
        raise DataQualityError("roster failed data-quality checks:\n  - " + "\n  - ".join(problems))


def fetch_roster(season: str, timeout: int = 30) -> pd.DataFrame:
    """Rostered players (TEAM_ID != 0) with their current team, from nba_api. Validated."""
    from nba_api.stats.endpoints import commonallplayers

    df = commonallplayers.CommonAllPlayers(
        is_only_current_season=1, league_id="00", season=season, timeout=timeout
    ).get_data_frames()[0]
    df = df.loc[df["TEAM_ID"] != 0, ROSTER_COLUMNS].reset_index(drop=True)
    validate_roster(df)
    return df

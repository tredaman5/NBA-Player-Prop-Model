"""
opponent_stats.py

Team-level opponent defense context, shared by the points/rebounds/assists
build_dataset.py modules so the logic isn't tripled.

Produces one row per (TEAM_ABBREVIATION, GAME_DATE) with rolling
"how much this team has allowed" stats. Values are shifted so a team's
entry for a given game only reflects games strictly before that date --
same no-leakage pattern as the player-level rolling features.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_league_team_gamelogs(
    season: str,
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_sec: float = 0.8,
) -> pd.DataFrame:
    """
    Pull all TEAM game logs for a season (league-wide) in one shot.

    Returns columns including:
    GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, MATCHUP, PTS, REB, AST, etc.
    """
    _ensure_dir(DATA_RAW)
    cache_path = DATA_RAW / f"league_team_gamelog_{season}_{season_type.replace(' ', '')}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    time.sleep(sleep_sec)
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",  # T = Team logs
    )
    df = resp.get_data_frames()[0].copy()

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df.to_parquet(cache_path, index=False)
    return df


def build_opponent_defense_table(
    team_gamelog: pd.DataFrame,
    stat_cols: Iterable[str] = ("PTS", "REB", "AST"),
    windows: Iterable[int] = (5, 10),
) -> pd.DataFrame:
    """
    For each team-game, find what the opponent scored in that same game
    (i.e. what this team allowed), then compute rolling allowed-stats per
    team, shifted so game N's value only uses games before N.

    Returns one row per (TEAM_ABBREVIATION, GAME_DATE, GAME_ID) with columns
    like OPP_PTS_ALLOWED_ROLL_5, OPP_REB_ALLOWED_ROLL_10, etc.
    """
    stat_cols = list(stat_cols)
    windows = list(windows)

    df = team_gamelog[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"] + stat_cols].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Every GAME_ID has exactly two team rows (home + away). Self-join on
    # GAME_ID and drop the row that matched itself to pair each team with
    # its opponent's same-game stat line.
    paired = df.merge(df, on="GAME_ID", suffixes=("", "_OPP"))
    paired = paired[paired["TEAM_ABBREVIATION"] != paired["TEAM_ABBREVIATION_OPP"]].copy()

    # "What TEAM_ABBREVIATION allowed" = what the opponent scored.
    allowed_cols = {c: f"{c}_ALLOWED" for c in stat_cols}
    for c in stat_cols:
        paired[allowed_cols[c]] = paired[f"{c}_OPP"]

    keep = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"] + list(allowed_cols.values())
    allowed = paired[keep].sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()

    for c in stat_cols:
        acol = allowed_cols[c]
        for w in windows:
            allowed[f"OPP_{c}_ALLOWED_ROLL_{w}"] = (
                allowed.groupby("TEAM_ABBREVIATION")[acol]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    out_cols = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"] + [
        f"OPP_{c}_ALLOWED_ROLL_{w}" for c in stat_cols for w in windows
    ]
    return allowed[out_cols].copy()


def opponent_feature_cols(stat: str, windows: Iterable[int] = (5, 10)) -> list[str]:
    """Convenience: the OPP_*_ALLOWED_ROLL_* column names for one stat (PTS/REB/AST)."""
    return [f"OPP_{stat}_ALLOWED_ROLL_{w}" for w in windows]


def merge_opponent_features(
    player_df: pd.DataFrame,
    defense_table: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Attach opponent-allowed rolling features to a player-game dataset.

    Matches each player row's OPP_ABBREVIATION + GAME_ID to the opponent
    team's own row in defense_table (built by build_opponent_defense_table),
    so a player facing team X gets team X's rolling allowed-stats.
    """
    feature_cols = list(feature_cols)
    right = defense_table[["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols].rename(
        columns={"TEAM_ABBREVIATION": "_OPP_TEAM_ABBR"}
    )

    out = player_df.merge(
        right,
        left_on=["GAME_ID", "OPP_ABBREVIATION"],
        right_on=["GAME_ID", "_OPP_TEAM_ABBR"],
        how="left",
    )
    return out.drop(columns=["_OPP_TEAM_ABBR"])

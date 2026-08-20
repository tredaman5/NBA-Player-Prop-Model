"""
injury_context.py

Official pregame inactive-player lists, used to build a "rotation teammates
out tonight" feature for the minutes model. A star teammate being ruled out
is one of the biggest drivers of another player's minutes swinging up or
down, and nothing else in this pipeline currently knows about it.

Data source: nba_api's boxscoresummaryv2 endpoint has an "InactivePlayers"
result set per game -- the real, official pregame inactive list, not an
inferred proxy. Unlike the rest of this pipeline (one league-wide call per
season), this endpoint is per-game, so building it for a full season is
~1,200+ API calls. It's cached to a single parquet afterward, so this cost
is paid once per season, not on every run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_game_inactive_players(
    game_ids: Iterable[str],
    season: str,
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_sec: float = 0.6,
    progress_every: int = 100,
) -> pd.DataFrame:
    """
    Fetch the official inactive-player list for each game_id.

    Returns columns: GAME_ID, TEAM_ABBREVIATION, PLAYER_ID, PLAYER_NAME
    (one row per inactive player per game). Games that fail to fetch are
    skipped (not the whole run) and reported at the end.
    """
    _ensure_dir(DATA_RAW)
    cache_path = DATA_RAW / f"game_inactive_players_{season}_{season_type.replace(' ', '')}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    game_ids = list(dict.fromkeys(game_ids))  # de-dupe, keep order
    rows = []
    failed = []

    print(f"[injury_context] Fetching inactive-player lists for {len(game_ids)} games...")
    for i, game_id in enumerate(game_ids, 1):
        try:
            time.sleep(sleep_sec)
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
            raw = summary.get_dict()
            names = [rs["name"] for rs in raw["resultSets"]]
            idx = names.index("InactivePlayers")
            df = summary.get_data_frames()[idx]

            if not df.empty:
                sub = df[["PLAYER_ID", "FIRST_NAME", "LAST_NAME", "TEAM_ABBREVIATION"]].copy()
                sub["GAME_ID"] = game_id
                sub["PLAYER_NAME"] = sub["FIRST_NAME"] + " " + sub["LAST_NAME"]
                rows.append(sub[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME"]])
        except Exception as e:
            failed.append((game_id, str(e)))

        if i % progress_every == 0:
            print(f"[injury_context]   {i}/{len(game_ids)} games processed ({len(failed)} failed so far)")

    if failed:
        print(f"[injury_context] WARNING: {len(failed)} games failed and were skipped.")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME"]
    )
    out.to_parquet(cache_path, index=False)
    print(f"[injury_context] Saved -> {cache_path} ({len(out)} inactive-player rows)")
    return out


def build_teammates_out_feature(
    player_minutes_df: pd.DataFrame,
    inactive_df: pd.DataFrame,
    min_threshold: float = 15.0,
) -> pd.DataFrame:
    """
    For each team-game, count how many players from that TEAM's most recent
    PRIOR game (who played >= min_threshold minutes in that prior game) are
    on tonight's official inactive list.

    Uses only the prior game's box score + tonight's pregame inactive
    announcement -- both legitimately known before tonight's game starts,
    so this is not leakage.

    Returns one row per (TEAM_ABBREVIATION, GAME_ID, GAME_DATE) with a
    TEAMMATES_OUT_ROTATION column.
    """
    df = player_minutes_df[["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    # One row per team-game with the set of rotation player_ids that game.
    team_games = (
        df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"])
        .apply(lambda g: frozenset(g.loc[g["MIN"] >= min_threshold, "PLAYER_ID"]), include_groups=False)
        .reset_index(name="ROTATION_PLAYER_IDS")
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    )

    # Shift so each game sees the PRIOR game's rotation set for that team.
    team_games["PRIOR_ROTATION_PLAYER_IDS"] = (
        team_games.groupby("TEAM_ABBREVIATION")["ROTATION_PLAYER_IDS"].shift(1)
    )

    inactive_by_team_game = (
        inactive_df.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["PLAYER_ID"]
        .apply(set)
        .reset_index(name="INACTIVE_PLAYER_IDS")
    )

    out = team_games.merge(inactive_by_team_game, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    def _count_overlap(row) -> int:
        prior = row["PRIOR_ROTATION_PLAYER_IDS"]
        inactive = row["INACTIVE_PLAYER_IDS"]
        if not isinstance(prior, frozenset) or not isinstance(inactive, set):
            return 0
        return len(prior & inactive)

    out["TEAMMATES_OUT_ROTATION"] = out.apply(_count_overlap, axis=1)

    return out[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "TEAMMATES_OUT_ROTATION"]]

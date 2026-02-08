"""
nba_data.py

Centralized data access layer for NBA stats using nba_api.
- Pulls player game logs for a season
- (Optional) pulls traditional box score for a game
- Caches responses to data/raw so you don't hammer endpoints

Notes:
- nba_api can be rate-limited; caching + small sleep helps.
- Season format: "2023-24", "2024-25", etc.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd

from nba_api.stats.endpoints import playergamelog, boxscoretraditionalv2
from nba_api.stats.static import players


# ---------- Paths ----------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../src/common -> repo root
DATA_RAW = REPO_ROOT / "data" / "raw"


# ---------- Utilities ----------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def season_str(season_start_year: int) -> str:
    """
    Convert a start year into NBA season string.
    Example: 2023 -> "2023-24"
    """
    yy = str((season_start_year + 1) % 100).zfill(2)
    return f"{season_start_year}-{yy}"


def _cache_path(name: str) -> Path:
    ensure_dir(DATA_RAW)
    return DATA_RAW / name


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


# ---------- Player helpers ----------

def get_player_id(full_name: str) -> int:
    """
    Return NBA player_id from a full name. Raises ValueError if not found/ambiguous.
    """
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No NBA player found for name='{full_name}'")

    # Try exact match first
    exact = [m for m in matches if m["full_name"].lower() == full_name.lower()]
    if len(exact) == 1:
        return int(exact[0]["id"])

    # If multiple, prefer active players
    active = [m for m in matches if m.get("is_active")]
    if len(active) == 1:
        return int(active[0]["id"])

    # Otherwise ambiguous
    raise ValueError(f"Ambiguous player name='{full_name}'. Matches={matches}")


# ---------- Data pulls ----------

def fetch_player_game_log(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """
    Fetch a single player's game log for a given season.

    Returns DataFrame with nba_api columns:
    GAME_ID, GAME_DATE, MATCHUP, MIN, PTS, REB, AST, FGA, FTA, FG3A, etc.
    """
    cache_file = _cache_path(f"player_gamelog_{player_id}_{season}_{season_type.replace(' ', '')}.parquet")
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    # rate-limit friendliness
    time.sleep(sleep_sec)

    resp = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = resp.get_data_frames()[0]

    # normalize types
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    _write_cache(df, cache_file)
    return df


def fetch_many_players_game_logs(
    player_ids: List[int],
    season: str,
    season_type: str = "Regular Season",
    use_cache: bool = True,
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """
    Fetch and concatenate game logs for many players.
    """
    frames = []
    for pid in player_ids:
        try:
            df = fetch_player_game_log(
                player_id=pid,
                season=season,
                season_type=season_type,
                use_cache=use_cache,
                sleep_sec=sleep_sec,
            )
            df = df.copy()
            df["PLAYER_ID"] = pid
            frames.append(df)
        except Exception as e:
            # Don't crash whole pipeline on one player
            print(f"[WARN] Failed player_id={pid}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


def fetch_boxscore_traditional(
    game_id: str,
    use_cache: bool = True,
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """
    Fetch traditional box score (player-level) for a single game.

    Useful when you want team context for a player game (lineup proxy, etc.).
    """
    cache_file = _cache_path(f"boxscore_traditional_{game_id}.parquet")
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    time.sleep(sleep_sec)

    resp = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    # Typically [player_stats_df, team_stats_df, ...]
    player_stats_df = resp.get_data_frames()[0]

    _write_cache(player_stats_df, cache_file)
    return player_stats_df


# ---------- Convenience: active player ids ----------

def get_active_player_ids() -> List[int]:
    """
    Return list of active NBA player ids (as of nba_api static dataset).
    """
    plist = players.get_active_players()
    return [int(p["id"]) for p in plist]

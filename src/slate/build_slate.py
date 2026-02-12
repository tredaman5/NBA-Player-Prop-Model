"""
build_slate.py (Odds API version)

Build slate feature tables for upcoming games using The Odds API events endpoint
instead of nba_api scoreboard.

Why:
- nba_api scoreboard can fail for future dates.
- The Odds API already provides upcoming events (games) and matches your odds fetch pipeline.

Inputs:
- prop datasets (historical): data/processed/<prop>_dataset_<season>_<seasontype>.parquet

Outputs:
- data/processed/slate_<prop>_features_<YYYY-MM-DD>.parquet

Environment:
- ODDS_API_KEY must be set
"""

from __future__ import annotations

import argparse
import os
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def _st(season_type: str) -> str:
    return season_type.replace(" ", "")


def _dataset_path(prop: str, season: str, season_type: str) -> Path:
    return PROCESSED / f"{prop}_dataset_{season}_{_st(season_type)}.parquet"


def _out_path(prop: str, date_str: str) -> Path:
    return PROCESSED / f"slate_{prop}_features_{date_str}.parquet"


def _require_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY env var. Set it: $env:ODDS_API_KEY='YOUR_KEY'")
    return key


def _get_json(url: str, params: dict):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_oddsapi_events(date_utc: str) -> List[dict]:
    """
    Fetch upcoming NBA events and filter to target UTC date (YYYY-MM-DD).
    """
    api_key = _require_api_key()
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events"
    events = _get_json(url, {"apiKey": api_key})

    target = dt.date.fromisoformat(date_utc)
    out = []
    for e in events:
        # commence_time is UTC ISO string
        d = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).date()
        if d == target:
            out.append(e)
    return out


def _latest_player_rows(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    For each PLAYER_ID, keep the latest row strictly before cutoff_date.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    cutoff = pd.to_datetime(cutoff_date)

    df = df[df["GAME_DATE"] < cutoff]
    if df.empty:
        return df

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    return df.groupby("PLAYER_ID", as_index=False).tail(1)


def _build_team_player_pool(prop_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Uses TEAM_ABBREVIATION (more portable across APIs than TEAM_ID).
    """
    required = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"]
    missing = [c for c in required if c not in prop_df.columns]
    if missing:
        raise ValueError(
            f"Prop dataset missing required columns: {missing}\n"
            "Your build_dataset should include TEAM_ABBREVIATION."
        )

    latest = _latest_player_rows(prop_df, cutoff_date=cutoff_date)
    return latest


def _attach_event_context(
    player_latest: pd.DataFrame,
    events: List[dict],
    date_str: str,
) -> pd.DataFrame:
    """
    Match players to events based on TEAM_ABBREVIATION.

    The Odds API uses full team names (e.g., 'Los Angeles Lakers').
    Your dataset uses abbreviations (e.g., 'LAL').

    We map Odds API team names -> abbreviations using a built-in dictionary.
    """
    target_date = pd.to_datetime(date_str)

    # Minimal NBA team name -> abbreviation map (can expand as needed)
    name_to_abbr = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "LA Clippers": "LAC",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    # Build map: team_abbr -> (event_id, is_home)
    team_to_event: Dict[str, Tuple[str, int]] = {}

    skipped = 0
    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        eid = e.get("id")

        if not home or not away or not eid:
            skipped += 1
            continue

        home_abbr = name_to_abbr.get(home)
        away_abbr = name_to_abbr.get(away)

        if not home_abbr or not away_abbr:
            skipped += 1
            continue

        team_to_event[home_abbr] = (eid, 1)
        team_to_event[away_abbr] = (eid, 0)

    if skipped:
        print(f"[build_slate] Skipped {skipped} events due to missing/unknown team names")

    df = player_latest.copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.strip()

    # Keep players whose teams are playing
    df = df[df["TEAM_ABBREVIATION"].isin(team_to_event.keys())].copy()
    if df.empty:
        return df

    df["EVENT_ID"] = df["TEAM_ABBREVIATION"].map(lambda ab: team_to_event[ab][0])
    df["IS_HOME"] = df["TEAM_ABBREVIATION"].map(lambda ab: team_to_event[ab][1])

    df["GAME_DATE_PREV"] = pd.to_datetime(df["GAME_DATE"])
    df["GAME_DATE"] = target_date

    df["DAYS_REST"] = (df["GAME_DATE"] - df["GAME_DATE_PREV"]).dt.days.clip(lower=0, upper=14)
    df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)

    return df


def build_slate(date_str: str, season: str, season_type: str, prop: str) -> Path:
    events = fetch_oddsapi_events(date_utc=date_str)
    if not events:
        raise RuntimeError(f"No Odds API events found for date {date_str}. (Check UTC date)")

    ds_path = _dataset_path(prop, season, season_type)
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset: {ds_path}\nRun: python -m src.props.{prop}.build_dataset")

    prop_df = pd.read_parquet(ds_path)
    latest = _build_team_player_pool(prop_df, cutoff_date=date_str)
    if latest.empty:
        raise RuntimeError(f"No historical rows before {date_str} in {ds_path}")

    slate = _attach_event_context(latest, events, date_str=date_str)
    if slate.empty:
        raise RuntimeError(
            "Slate is empty after mapping teams to events.\n"
            "Likely causes:\n"
            "- TEAM_ABBREVIATION missing or inconsistent\n"
            "- Team name mapping missing a team string from Odds API\n"
        )

    out = _out_path(prop, date_str)
    slate.to_parquet(out, index=False)
    print(f"Saved slate features -> {out}")
    print(f"Rows: {len(slate)} | Events: {len(events)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC date to build slate for)")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--prop", choices=["points", "rebounds", "assists", "all"], default="all")
    args = ap.parse_args()

    props = ["points", "rebounds", "assists"] if args.prop == "all" else [args.prop]
    for p in props:
        build_slate(date_str=args.date, season=args.season, season_type=args.season_type, prop=p)


if __name__ == "__main__":
    main()
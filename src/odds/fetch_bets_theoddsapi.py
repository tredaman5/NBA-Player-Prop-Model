"""
fetch_bets_theoddsapi.py

Fetch NBA player props (points/rebounds/assists) from The Odds API
and write a user-friendly CSV compatible with slip_from_csv.py.

KEY FEATURE:
- Filters to ONLY players that exist in your model prediction parquets
  so you don't get "no match" rows later.

Requires:
  pip install requests
Environment:
  $env:ODDS_API_KEY="YOUR_KEY"

Output columns:
prop,season,season_type,player_name,game_date,line,odds_over,odds_under,kelly_multiplier,min_edge
(+ debug columns: best_over_book, best_under_book, event_id)
"""

from __future__ import annotations

import os
import time
import datetime as dt
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

DEFAULT_MARKETS = ["player_points", "player_rebounds", "player_assists"]
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"

# Keep short to avoid huge CSVs + reduce "noise"
DEFAULT_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars"]


@dataclass
class Outcome:
    side: str        # "Over" or "Under"
    player: str      # API outcome.description (player name)
    line: float      # outcome.point
    price: int       # outcome.price (american)
    book: str        # bookmaker.key


# ----------------------------
# Helpers
# ----------------------------

def _require_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY env var. Set it first: $env:ODDS_API_KEY='YOUR_KEY'")
    return key


def _get_json(url: str, params: dict, sleep_s: float = 0.25):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(sleep_s)  # gentle throttle
    return r.json()


def _iso_utc_to_date(iso: str) -> str:
    # e.g. "2026-02-07T01:00:00Z" -> "2026-02-07"
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date().isoformat()


def normalize_name(name: str) -> str:
    """
    Normalize names for matching across sources:
    - lowercase
    - strip extra spaces
    - remove accents (Jokić -> Jokic)
    - remove periods
    """
    s = str(name).strip().lower()
    s = " ".join(s.split())
    s = s.replace(".", "")

    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def repo_root() -> Path:
    # src/odds/fetch_bets_theoddsapi.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def load_model_name_map(season: str, season_type: str) -> Dict[str, str]:
    """
    Build a mapping:
      normalized_name -> model_exact_name

    We load from your *_with_pred parquets because they contain the exact PLAYER_NAME
    strings your betslip matcher expects.
    """
    root = repo_root()
    st = season_type.replace(" ", "")
    processed = root / "data" / "processed"

    paths = [
        processed / f"points_with_pred_{season}_{st}.parquet",
        processed / f"rebounds_with_pred_{season}_{st}.parquet",
        processed / f"assists_with_pred_{season}_{st}.parquet",
    ]

    name_map: Dict[str, str] = {}
    found_any = False

    for p in paths:
        if not p.exists():
            continue
        found_any = True
        df = pd.read_parquet(p, columns=["PLAYER_NAME"]).copy()
        for nm in df["PLAYER_NAME"].dropna().astype(str).unique():
            key = normalize_name(nm)
            # Keep first seen model spelling (good enough)
            if key not in name_map:
                name_map[key] = nm

    if not found_any:
        raise FileNotFoundError(
            "Could not find any *_with_pred parquet files to build the player filter.\n"
            "Run:\n"
            "  python main.py predict-prop --prop points --season <season>\n"
            "  python main.py predict-prop --prop rebounds --season <season>\n"
            "  python main.py predict-prop --prop assists --season <season>\n"
        )

    return name_map


# ----------------------------
# The Odds API calls
# ----------------------------

def fetch_events(api_key: str, date_utc: Optional[str] = None) -> List[dict]:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events"
    events = _get_json(url, {"apiKey": api_key})

    if not date_utc:
        return events

    target = dt.date.fromisoformat(date_utc)
    out = []
    for e in events:
        d = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).date()
        if d == target:
            out.append(e)
    return out


def fetch_event_odds(api_key: str, event_id: str, markets: List[str], regions: str, odds_format: str) -> dict:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
    }
    return _get_json(url, params)


def parse_outcomes(event_odds: dict, allowed_books: Optional[List[str]] = None) -> Dict[str, List[Outcome]]:
    out: Dict[str, List[Outcome]] = {}

    for bm in event_odds.get("bookmakers", []):
        book_key = bm.get("key")
        if allowed_books and book_key not in allowed_books:
            continue

        for m in bm.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                side = o.get("name")
                player = o.get("description")
                line = o.get("point")
                price = o.get("price")

                if side not in ("Over", "Under"):
                    continue
                if player is None or line is None or price is None:
                    continue

                out.setdefault(mkey, []).append(
                    Outcome(side=side, player=str(player).strip(), line=float(line), price=int(price), book=str(book_key))
                )

    return out


def best_over_under(outcomes: List[Outcome]) -> Dict[Tuple[str, float], Dict[str, Outcome]]:
    """
    Group by (player, line), choose best price for Over and Under across books.
    Higher American price is better payout (e.g. -105 better than -120).
    """
    grouped: Dict[Tuple[str, float], Dict[str, Outcome]] = {}

    for o in outcomes:
        key = (o.player, o.line)
        grouped.setdefault(key, {})
        cur = grouped[key].get(o.side)
        if cur is None or o.price > cur.price:
            grouped[key][o.side] = o

    return grouped


def market_to_prop(market_key: str) -> Optional[str]:
    return {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
    }.get(market_key)


# ----------------------------
# Main
# ----------------------------

def main(
    out_path: str = "bets_today.csv",
    date_utc: Optional[str] = None,          # YYYY-MM-DD UTC; if None, includes upcoming
    season: str = "2025-26",
    season_type: str = "Regular Season",
    regions: str = DEFAULT_REGIONS,
    markets: Optional[List[str]] = None,
    books: Optional[List[str]] = None,
    kelly_multiplier: float = 0.25,
    min_edge: float = 0.02,
) -> None:
    api_key = _require_api_key()
    markets = markets or DEFAULT_MARKETS
    books = books or DEFAULT_BOOKMAKERS

    # Build player filter from YOUR model outputs
    model_name_map = load_model_name_map(season=season, season_type=season_type)
    model_name_keys = set(model_name_map.keys())

    events = fetch_events(api_key, date_utc=date_utc)
    rows: List[dict] = []

    for e in events:
        event_id = e["id"]
        game_date = _iso_utc_to_date(e["commence_time"])

        event_odds = fetch_event_odds(api_key, event_id, markets=markets, regions=regions, odds_format=DEFAULT_ODDS_FORMAT)
        parsed = parse_outcomes(event_odds, allowed_books=books)

        for mkey, outcomes in parsed.items():
            prop = market_to_prop(mkey)
            if not prop:
                continue

            best = best_over_under(outcomes)

            for (api_player, line), sides in best.items():
                if "Over" not in sides or "Under" not in sides:
                    continue

                # FILTER HERE: keep only players present in YOUR model
                api_key_name = normalize_name(api_player)
                if api_key_name not in model_name_keys:
                    continue

                # Use model's exact spelling so slip_from_csv matches perfectly
                player_name = model_name_map[api_key_name]

                over = sides["Over"]
                under = sides["Under"]

                rows.append(
                    {
                        "prop": prop,
                        "season": season,
                        "season_type": season_type,
                        "player_name": player_name,   # model exact
                        "game_date": game_date,
                        "line": float(line),
                        "odds_over": int(over.price),
                        "odds_under": int(under.price),
                        "kelly_multiplier": float(kelly_multiplier),
                        "min_edge": float(min_edge),
                        # Debug fields (optional; keep or delete)
                        "best_over_book": over.book,
                        "best_under_book": under.book,
                        "event_id": event_id,
                    }
                )

    if rows:
        df = pd.DataFrame(rows).sort_values(["game_date", "prop", "player_name", "line"])
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path} ({len(df)} rows)")
    else:
        print(
            "No rows produced.\n"
            "Possible causes:\n"
            "- No events returned for that date\n"
            "- Books list too strict\n"
            "- Markets not returned by API for events\n"
            "- Player names not matching (should be rare now)\n"
        )


if __name__ == "__main__":
    # Example: today's UTC date filter
    # today_utc = dt.datetime.now(dt.timezone.utc).date().isoformat()
    # main(out_path="bets_today.csv", date_utc=today_utc)

    main(out_path="bets_today.csv", date_utc=None)

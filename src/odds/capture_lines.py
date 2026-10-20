"""
capture_lines.py

Append-only capture of raw sportsbook player-prop lines, independent of the
model. Sportsbook lines are the one input we can't rebuild later (historical
odds are paid); projections can always be regenerated from nba_api data.

Built for The Odds API's free tier (500 credits/month). Cost is one credit per
market per region per game, so three markets in the US region is ~3 credits a
game and a full nightly slate is ~20+ credits. To stay inside the budget:

- only games tipping off within --within-minutes are captured (lines near
  tip-off are also the closest thing to a closing line we can get),
- a game already captured recently is skipped, so re-running costs nothing,
- a per-run credit cap and a reserve floor stop the run before the month's
  quota is gone,
- --dry-run lists what would be captured and what it would cost, spending
  nothing (the events listing is not charged against the quota).

Output: data/raw/odds_lines_log.csv (gitignored -- keep it private and back it
up; the provider's terms bar redistributing their data as files or feeds).
One row per (capture, bookmaker, market, player, line) with over/under prices.
The log schema is provider-agnostic; only this file knows about The Odds API.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.odds.fetch_bets_theoddsapi import BASE_URL, DEFAULT_MARKETS, SPORT_KEY, parse_outcomes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "raw" / "odds_lines_log.csv"

REGION = "us"

LOG_COLUMNS = [
    "captured_at",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market",
    "player",
    "line",
    "over_price",
    "under_price",
]

# (url, params) -> (parsed json, response headers)
HttpGet = Callable[[str, dict], Tuple[object, Dict[str, str]]]


class OddsApiError(RuntimeError):
    """Raised for any API failure. Messages never contain the API key."""


def _require_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY env var. Set it in your shell before running.")
    return key


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def _default_http_get(url: str, params: dict) -> Tuple[object, Dict[str, str]]:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        hint = {401: " (invalid API key)", 429: " (rate limited or quota exhausted)"}.get(r.status_code, "")
        raise OddsApiError(f"HTTP {r.status_code}{hint}: {r.text[:200]}")
    return r.json(), dict(r.headers)


def _header_int(headers: Dict[str, str], name: str) -> Optional[int]:
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    try:
        return int(float(lowered[name]))
    except (KeyError, TypeError, ValueError):
        return None


def _parse_time(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _recently_captured_event_ids(log_path: Path, now: datetime, min_gap_minutes: int) -> set:
    if min_gap_minutes <= 0 or not log_path.exists():
        return set()
    df = pd.read_csv(log_path, usecols=["event_id", "captured_at"])
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    cutoff = pd.Timestamp(now - timedelta(minutes=min_gap_minutes))
    return set(df.loc[df["captured_at"] >= cutoff, "event_id"].astype(str))


def _event_rows(event: dict, event_odds: dict, captured_at: str) -> List[dict]:
    """Flatten one event's odds into one row per (bookmaker, market, player, line)."""
    rows = []
    for market, outcomes in parse_outcomes(event_odds).items():
        grouped: Dict[Tuple[str, str, float], Dict[str, int]] = {}
        for o in outcomes:
            grouped.setdefault((o.book, o.player, o.line), {})[o.side] = o.price
        for (book, player, line), sides in grouped.items():
            rows.append(
                {
                    "captured_at": captured_at,
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": book,
                    "market": market,
                    "player": player,
                    "line": line,
                    "over_price": sides.get("Over"),
                    "under_price": sides.get("Under"),
                }
            )
    return rows


def capture_lines(
    api_key: str,
    log_path: Path = DEFAULT_LOG_PATH,
    markets: Sequence[str] = tuple(DEFAULT_MARKETS),
    within_minutes: int = 120,
    max_credits: int = 60,
    reserve: int = 25,
    min_gap_minutes: int = 360,
    dry_run: bool = False,
    http_get: Optional[HttpGet] = None,
    now: Optional[datetime] = None,
    sleep_sec: float = 0.25,
) -> dict:
    """
    Capture lines for upcoming games and append them to the log. Returns a
    summary dict. Raises OddsApiError (key redacted) on API failure.
    """
    http_get = http_get or _default_http_get
    fixed_now = now  # injected by tests; live runs stamp each capture with the real time
    now = now or datetime.now(timezone.utc)
    cost_per_event = len(markets) * 1  # one region
    base = f"{BASE_URL}/sports/{SPORT_KEY}"

    def call(url: str, params: dict) -> Tuple[object, Dict[str, str]]:
        try:
            return http_get(url, {**params, "apiKey": api_key})
        except Exception as e:  # never let the key reach a log or traceback
            raise OddsApiError(_redact(str(e), api_key)) from None

    events, headers = call(f"{base}/events", {})
    remaining = _header_int(headers, "x-requests-remaining")

    recent = _recently_captured_event_ids(log_path, now, min_gap_minutes)
    window_end = now + timedelta(minutes=within_minutes)
    eligible = sorted(
        (
            e
            for e in events
            if now < _parse_time(e["commence_time"]) <= window_end and str(e["id"]) not in recent
        ),
        key=lambda e: e["commence_time"],
    )

    summary = {
        "events_listed": len(events),
        "events_eligible": len(eligible),
        "events_captured": 0,
        "rows_written": 0,
        "credits_spent": 0,
        "credits_remaining": remaining,
        "estimated_cost": len(eligible) * cost_per_event,
        "stopped_reason": None,
    }

    if dry_run:
        summary["stopped_reason"] = "dry_run"
        return summary

    all_rows: List[dict] = []
    for event in eligible:
        if summary["credits_spent"] + cost_per_event > max_credits:
            summary["stopped_reason"] = "max_credits"
            break
        if remaining is not None and remaining - cost_per_event < reserve:
            summary["stopped_reason"] = "reserve"
            break

        event_odds, odds_headers = call(
            f"{base}/events/{event['id']}/odds",
            {"regions": REGION, "markets": ",".join(markets), "oddsFormat": "american"},
        )
        captured_at = (fixed_now or datetime.now(timezone.utc)).isoformat()
        all_rows.extend(_event_rows(event, event_odds, captured_at))

        remaining = _header_int(odds_headers, "x-requests-remaining")
        last_cost = _header_int(odds_headers, "x-requests-last")
        summary["credits_spent"] += last_cost if last_cost is not None else cost_per_event
        summary["events_captured"] += 1
        if sleep_sec:
            time.sleep(sleep_sec)

    summary["credits_remaining"] = remaining
    if all_rows:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows, columns=LOG_COLUMNS).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False
        )
    summary["rows_written"] = len(all_rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture raw player-prop lines (free-tier friendly).")
    ap.add_argument("--within-minutes", type=int, default=120, help="Only games tipping off within this many minutes.")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated market keys.")
    ap.add_argument("--max-credits", type=int, default=60, help="Hard cap on credits spent this run.")
    ap.add_argument("--reserve", type=int, default=25, help="Stop if remaining quota would drop below this.")
    ap.add_argument("--min-gap-minutes", type=int, default=360, help="Skip games captured this recently (0 = allow).")
    ap.add_argument("--dry-run", action="store_true", help="List what would be captured; spends no credits.")
    ap.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = ap.parse_args()

    try:
        summary = capture_lines(
            api_key=_require_api_key(),
            log_path=args.log_path,
            markets=[m.strip() for m in args.markets.split(",") if m.strip()],
            within_minutes=args.within_minutes,
            max_credits=args.max_credits,
            reserve=args.reserve,
            min_gap_minutes=args.min_gap_minutes,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:  # includes OddsApiError; messages never contain the key
        sys.exit(f"error: {e}")

    print(f"Games listed:      {summary['events_listed']}")
    print(f"Eligible now:      {summary['events_eligible']} (est. {summary['estimated_cost']} credits)")
    if args.dry_run:
        print("Dry run -- nothing captured, no credits spent.")
    else:
        print(f"Captured:          {summary['events_captured']} games, {summary['rows_written']} rows")
        print(f"Credits spent:     {summary['credits_spent']}")
        if summary["stopped_reason"]:
            print(f"Stopped early:     {summary['stopped_reason']}")
    print(f"Credits remaining: {summary['credits_remaining'] if summary['credits_remaining'] is not None else 'unknown'}")


if __name__ == "__main__":
    main()

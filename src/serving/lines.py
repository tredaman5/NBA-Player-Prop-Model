"""
lines.py: turn the captured sportsbook log into one consensus line per player.

Books post several alternate lines per player (Bovada especially), so each book's MAIN line is the
one priced closest to even odds; the consensus is the median of those across books. Only each
game's latest capture is used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LOG_COLUMNS = [
    "captured_at", "event_id", "commence_time", "home_team", "away_team",
    "bookmaker", "market", "player", "line", "over_price", "under_price",
]
MIN_BOOKS = 3


def _implied(american: pd.Series) -> pd.Series:
    """American odds to implied probability (vig not removed)."""
    o = american.astype(float)
    return (100 / (o + 100)).where(o > 0, -o / (-o + 100))


def load_line_log(path) -> pd.DataFrame:
    """Read the captured log with timezone-aware UTC timestamps."""
    log = pd.read_csv(Path(path))
    missing = [c for c in LOG_COLUMNS if c not in log.columns]
    if missing:
        raise ValueError(f"line log is missing columns {missing}")
    if log.empty:
        raise ValueError("line log is empty")
    for col in ("captured_at", "commence_time"):
        log[col] = pd.to_datetime(log[col], utc=True)
    return log


def main_lines(log: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, book, market, player): the line whose two sides are priced closest to even."""
    priced = log.dropna(subset=["over_price", "under_price"]).copy()
    priced["_skew"] = (_implied(priced["over_price"]) - _implied(priced["under_price"])).abs()
    keys = ["event_id", "bookmaker", "market", "player"]
    return priced.sort_values("_skew").groupby(keys, as_index=False).first().drop(columns="_skew")


def consensus_lines(log: pd.DataFrame, min_books: int = MIN_BOOKS) -> pd.DataFrame:
    """Median main line across books, from each event's latest capture only."""
    latest = log[log["captured_at"] == log.groupby("event_id")["captured_at"].transform("max")]
    main = main_lines(latest)
    out = (
        main.groupby(["event_id", "market", "player"], as_index=False)
        .agg(
            line=("line", "median"),
            books=("bookmaker", "nunique"),
            captured_at=("captured_at", "first"),
            commence_time=("commence_time", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
    )
    out["thin_market"] = out["books"] < min_books
    return out

"""
names.py: link sportsbook player names to NBA player ids.

A name is only matched against players on the event's two teams, which handles trades and
duplicate names across teams, and gives a precise reason when it fails.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase, no accents, no periods or apostrophes, hyphens as spaces, no generational suffix."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(c)).lower()
    s = s.replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    return " ".join(t for t in s.split() if t not in _SUFFIXES)


def match_to_roster(lines: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """
    Add PLAYER_ID, TEAM_ID, TEAM_ABBREVIATION and match_status to `lines`.

    lines needs: player, home_abbr, away_abbr. roster needs: PERSON_ID, DISPLAY_FIRST_LAST,
    TEAM_ID, TEAM_ABBREVIATION. match_status is "matched", "not_on_either_roster" or "ambiguous".
    Every input row comes back exactly once, in order.
    """
    left = lines.reset_index(drop=True).copy()
    left["_row"] = range(len(left))
    left["_key"] = left["player"].map(normalize_name)

    people = roster[["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_ABBREVIATION"]].copy()
    people["_key"] = people["DISPLAY_FIRST_LAST"].map(normalize_name)

    pairs = left[["_row", "_key", "home_abbr", "away_abbr"]].merge(people, on="_key", how="inner")
    pairs = pairs[(pairs["TEAM_ABBREVIATION"] == pairs["home_abbr"]) | (pairs["TEAM_ABBREVIATION"] == pairs["away_abbr"])]
    counts = pairs.groupby("_row").size()

    unique = pairs[pairs["_row"].map(counts) == 1][["_row", "PERSON_ID", "TEAM_ID", "TEAM_ABBREVIATION"]]
    out = left.merge(unique, on="_row", how="left").rename(columns={"PERSON_ID": "PLAYER_ID"})
    n = out["_row"].map(counts).fillna(0)
    out["match_status"] = n.map(lambda c: "matched" if c == 1 else ("ambiguous" if c > 1 else "not_on_either_roster"))
    return out.drop(columns=["_row", "_key"]).reset_index(drop=True)

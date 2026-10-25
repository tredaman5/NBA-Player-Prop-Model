"""
Shared synthetic league for slate and pipeline tests: Boston (BOS, team id 100, players 1-2) and
Detroit (DET, team id 200, players 3-4) have played 12 games, last on 2025-01-23. Tonight is
2025-01-25, BOS vs DET at 6:30pm ET. The stand-in models are simple functions of the features.
"""

import datetime as dt

import numpy as np
import pandas as pd

from src.common.teams import TEAM_ABBR
from src.features.columns import MINUTES_FEATURES, PROP_FEATURES
from src.serving.lines import LOG_COLUMNS
from src.serving.models import Models
from tests.league_fixture import make_league

SLATE_DATE = dt.date(2025, 1, 25)
AS_OF = pd.Timestamp("2025-01-25T12:00:00Z")
TIP = "2025-01-25T23:30:00+00:00"  # 6:30pm ET


class Stub:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return np.asarray(self.fn(X), dtype=float)


def make_models():
    return Models(
        minutes=Stub(lambda X: X["MIN_ROLL_5"]),
        props={"points": Stub(lambda X: X["MIN_PRED"] * 0.5),
               "rebounds": Stub(lambda X: X["MIN_PRED"] * 0.2),
               "assists": Stub(lambda X: X["MIN_PRED"] * 0.1)},
        features={"minutes": MINUTES_FEATURES, **PROP_FEATURES},
        rmse={"points": 5.0, "rebounds": 2.0, "assists": 1.5},
        trained_through="2025-01-01",
        label="test",
    )


def make_history():
    return make_league(n_games=12, teams=("BOS", "DET"))


def make_roster(extra=()):
    rows = [(1, "P1", 100, "BOS"), (2, "P2", 100, "BOS"), (3, "P3", 200, "DET"), (4, "P4", 200, "DET"), *extra]
    others = [t for t in sorted(set(TEAM_ABBR.values())) if t not in ("BOS", "DET")]
    rows += [(1000 + i, f"Filler{i}", 300 + i, t) for i, t in enumerate(others)]
    return pd.DataFrame(rows, columns=["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_ABBREVIATION"])


def make_log(players=("P1", "P2", "P3", "P4"), books=("dk", "fd", "mgm"), line=10.5, market="player_points",
         home="Boston Celtics", away="Detroit Pistons", event="e1", tip=TIP):
    rows = [{"captured_at": "2025-01-25T20:00:00+00:00", "event_id": event, "commence_time": tip,
             "home_team": home, "away_team": away, "bookmaker": b, "market": market, "player": p,
             "line": line, "over_price": -110, "under_price": -110} for p in players for b in books]
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df

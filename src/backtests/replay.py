"""
replay.py: score every captured line against what happened.

Training and serving share one feature builder, so the features a player had going into a game are
exactly that player-game's row in the training frame: shift(1) means a row never sees its own box
score. The replay builds the frame once over the full history and looks each logged line up in it.
Every line gets a status; nothing is dropped silently, and only "scored" lines carry a projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtests.rules import FIRST_GAME_DATE, MIN_PRIOR_GAMES
from src.common.teams import TEAM_ABBR
from src.common.validation import validate_serving_frame
from src.features.builder import build_features
from src.features.columns import PROP_TARGETS
from src.serving.lines import MIN_BOOKS, consensus_lines
from src.serving.models import MARKET_TO_PROP, Models
from src.serving.names import match_to_roster, normalize_name
from src.serving.slate import EASTERN

REPLAY_COLUMNS = [
    "event_id", "game_date", "player", "player_id", "team", "market", "line", "books", "mins_before_tip",
    "be_over", "be_under", "projection", "min_pred", "prior_games", "actual", "status", "model_label",
]


def replay_lines(
    log: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    models: Models,
    min_books: int = MIN_BOOKS,
    min_prior: int = MIN_PRIOR_GAMES,
    first_game_date=FIRST_GAME_DATE,
) -> pd.DataFrame:
    cons = consensus_lines(log, min_books)
    cons = cons[cons["market"].isin(MARKET_TO_PROP)].reset_index(drop=True)
    cons["game_date"] = cons["commence_time"].dt.tz_convert(EASTERN).dt.date
    cons["home_abbr"] = cons["home_team"].map(TEAM_ABBR)
    cons["away_abbr"] = cons["away_team"].map(TEAM_ABBR)
    cons["mins_before_tip"] = (cons["commence_time"] - cons["captured_at"]).dt.total_seconds() / 60

    status = pd.Series([None] * len(cons), index=cons.index, dtype=object)

    def mark(mask, text):
        status[mask & status.isna()] = text

    trained_through = pd.Timestamp(models.trained_through).date()
    mark(cons["home_abbr"].isna() | cons["away_abbr"].isna(), "unknown team name")
    mark(cons["game_date"] < first_game_date, "before sample start")
    mark(cons["game_date"] <= trained_through, "in training window")
    mark(cons["thin_market"], "thin market")

    dates = players["GAME_DATE"].dt.date
    played = set(zip(dates, players["TEAM_ABBREVIATION"]))
    not_yet = [(d, h) not in played and (d, a) not in played
               for d, h, a in zip(cons["game_date"], cons["home_abbr"], cons["away_abbr"])]
    mark(pd.Series(not_yet, index=cons.index), "game not played yet")

    known = set(zip(players["PLAYER_NAME"].map(normalize_name), players["TEAM_ABBREVIATION"]))
    cons["player_id"] = np.nan
    cons["game_id"] = None
    cons["team"] = None
    todo = cons[status.isna()]
    for date, group in todo.groupby("game_date"):
        day = players[(dates == date) & (players["MIN"] > 0)]
        roster = pd.DataFrame({
            "PERSON_ID": day["PLAYER_ID"].to_numpy(), "DISPLAY_FIRST_LAST": day["PLAYER_NAME"].to_numpy(),
            "TEAM_ID": day["TEAM_ID"].to_numpy(), "TEAM_ABBREVIATION": day["TEAM_ABBREVIATION"].to_numpy(),
        })
        game_of = day.set_index("PLAYER_ID")["GAME_ID"]
        matched = match_to_roster(group, roster)
        matched.index = group.index
        for idx, row in matched.iterrows():
            if row["match_status"] == "matched":
                cons.loc[idx, "player_id"] = row["PLAYER_ID"]
                cons.loc[idx, "game_id"] = game_of[row["PLAYER_ID"]]
                cons.loc[idx, "team"] = row["TEAM_ABBREVIATION"]
            else:
                key = normalize_name(row["player"])
                knew = (key, row["home_abbr"]) in known or (key, row["away_abbr"]) in known
                status[idx] = "did not play" if knew and row["match_status"] == "not_on_either_roster" else "name not found"

    feat = build_features(players, teams)
    feat["prior_games"] = (
        feat.groupby("PLAYER_ID")["MIN"].transform(lambda m: (m > 0).cumsum() - (m > 0))
    )
    feat = feat.set_index(["GAME_ID", "PLAYER_ID"])

    matched_rows = cons[status.isna()].copy()
    keys = list(zip(matched_rows["game_id"], matched_rows["player_id"].astype(int)))
    rows = feat.loc[keys]
    cons.loc[matched_rows.index, "prior_games"] = rows["prior_games"].to_numpy()
    few = pd.Series(False, index=cons.index)
    few[matched_rows.index] = rows["prior_games"].to_numpy() < min_prior
    mark(few, "few prior games")

    scored = cons[status.isna()].copy()
    cons["projection"] = np.nan
    cons["min_pred"] = np.nan
    cons["actual"] = np.nan
    if len(scored):
        keys = list(zip(scored["game_id"], scored["player_id"].astype(int)))
        rows = feat.loc[keys].copy()
        validate_serving_frame(rows, models.features["minutes"])
        rows["MIN_PRED"] = np.clip(models.minutes.predict(rows[models.features["minutes"]].astype(float)), 0, 60)
        cons.loc[scored.index, "min_pred"] = rows["MIN_PRED"].to_numpy()
        for market, prop in MARKET_TO_PROP.items():
            sel = (scored["market"] == market).to_numpy()
            if not sel.any():
                continue
            part = rows[sel]
            validate_serving_frame(part, models.features[prop])
            pred = np.clip(models.props[prop].predict(part[models.features[prop]].astype(float)), 0, 80)
            cons.loc[scored.index[sel], "projection"] = pred
            cons.loc[scored.index[sel], "actual"] = part[PROP_TARGETS[prop]].to_numpy()
        status[scored.index] = "scored"

    cons["status"] = status
    cons["model_label"] = models.label
    out = cons[REPLAY_COLUMNS]
    return out.sort_values(["game_date", "event_id", "player"]).reset_index(drop=True)

"""
slate.py: tonight's projections next to the sportsbook lines.

For every player with a posted line: the model's projection, the median line across books, and the
difference. Nothing else: no edge, no EV, no probability, no pick.

Features come from the same build_features() training uses, with tonight's games as upcoming rows.
History is cut off at the slate date, so replaying a past date can never see that night's results.
Every player that does not make the slate is listed with a reason; nobody is dropped silently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.seasons import season_for_date
from src.common.teams import TEAM_ABBR
from src.common.validation import DataQualityError, validate_serving_frame
from src.features.builder import build_features
from src.serving.lines import MIN_BOOKS, consensus_lines
from src.serving.models import MARKET_TO_PROP, Models
from src.serving.names import match_to_roster

MIN_GAMES = 5          # prior games needed before a projection means anything
MAX_STALE_DAYS = 3     # newest game in the same season may not be older than this
EASTERN = "America/New_York"

SLATE_COLUMNS = [
    "event_id", "tip_et", "home", "away", "player", "player_id", "team", "opponent", "market",
    "line", "books", "projection", "difference", "min_pred", "prior_games", "captured_at", "mins_before_tip",
]
SKIPPED_COLUMNS = ["player", "event_id", "market", "reason"]


@dataclass
class SlateResult:
    slate: pd.DataFrame
    skipped: pd.DataFrame
    meta: Dict[str, object]


def _check_fresh(history_players: pd.DataFrame, slate_date: dt.date) -> pd.Timestamp:
    """Newest game before the slate date. Refuse a history that is stale within the same season."""
    if history_players.empty:
        raise DataQualityError("no game history before the slate date")
    last = history_players["GAME_DATE"].max()
    gap = (pd.Timestamp(slate_date) - last).days
    if season_for_date(last) == season_for_date(slate_date) and gap > MAX_STALE_DAYS:
        raise DataQualityError(
            f"history is stale: the newest game before {slate_date} was {last.date()}, {gap} days earlier"
        )
    return last


def _skip(rows: pd.DataFrame, reason) -> pd.DataFrame:
    out = rows[["player", "event_id", "market"]].copy()
    out["reason"] = reason
    return out


def build_slate(
    log: pd.DataFrame,
    roster: pd.DataFrame,
    history_players: pd.DataFrame,
    history_teams: pd.DataFrame,
    models: Models,
    as_of: pd.Timestamp,
    slate_date: dt.date,
    min_games: int = MIN_GAMES,
    min_books: int = MIN_BOOKS,
    include_started: bool = False,
) -> SlateResult:
    as_of = pd.Timestamp(as_of)
    missing_abbr = sorted(set(TEAM_ABBR.values()) - set(roster["TEAM_ABBREVIATION"]))
    if missing_abbr:
        raise DataQualityError(
            f"team abbreviations differ between the odds provider and the roster: {missing_abbr} are not on the roster"
        )

    day = pd.Timestamp(slate_date)
    hp = history_players[history_players["GAME_DATE"] < day]
    ht = history_teams[history_teams["GAME_DATE"] < day]
    last_date = _check_fresh(hp, slate_date)

    cons = consensus_lines(log, min_books)
    cons = cons[cons["market"].isin(MARKET_TO_PROP)].copy()
    cons["tip_et"] = cons["commence_time"].dt.tz_convert(EASTERN)
    cons = cons[cons["tip_et"].dt.date == slate_date]
    if not include_started:
        cons = cons[cons["commence_time"] > as_of]
    cons = cons.reset_index(drop=True)
    n_lines = len(cons)

    skipped: List[pd.DataFrame] = []
    cons["home_abbr"] = cons["home_team"].map(TEAM_ABBR)
    cons["away_abbr"] = cons["away_team"].map(TEAM_ABBR)

    unknown = cons["home_abbr"].isna() | cons["away_abbr"].isna()
    skipped.append(_skip(cons[unknown], "unknown team name"))
    cons = cons[~unknown]

    thin = cons["thin_market"]
    skipped.append(_skip(cons[thin], cons.loc[thin, "books"].map(lambda b: f"line from only {b} book(s)")))
    cons = cons[~thin]

    matched = match_to_roster(cons, roster)
    reasons = {"not_on_either_roster": "not on either team's roster", "ambiguous": "ambiguous name"}
    bad = matched["match_status"] != "matched"
    skipped.append(_skip(matched[bad], matched.loc[bad, "match_status"].map(reasons)))
    matched = matched[~bad].copy()
    matched["PLAYER_ID"] = matched["PLAYER_ID"].astype(int)

    prior = hp[hp["MIN"] > 0].groupby("PLAYER_ID").size()
    matched["prior_games"] = matched["PLAYER_ID"].map(prior).fillna(0).astype(int)
    short = matched["prior_games"] < min_games
    skipped.append(_skip(matched[short], matched.loc[short, "prior_games"].map(lambda n: f"only {n} prior games")))
    ok = matched[~short].copy()

    skipped_df = pd.concat(skipped, ignore_index=True).reindex(columns=SKIPPED_COLUMNS)
    meta = {
        "slate_date": str(slate_date),
        "as_of": str(as_of),
        "history_last_date": str(last_date.date()),
        "model_label": models.label,
        "model_trained_through": models.trained_through,
        "n_lines": n_lines,
        "n_projected": len(ok),
        "n_skipped": len(skipped_df),
    }
    if ok.empty:
        return SlateResult(pd.DataFrame(columns=SLATE_COLUMNS), skipped_df, meta)

    team_id = roster.drop_duplicates("TEAM_ABBREVIATION").set_index("TEAM_ABBREVIATION")["TEAM_ID"]
    events = ok.drop_duplicates("event_id")
    games = pd.DataFrame(
        {
            "GAME_ID": events["event_id"].to_numpy(),
            "GAME_DATE": day,
            "HOME_TEAM_ID": events["home_abbr"].map(team_id).to_numpy(),
            "HOME_ABBR": events["home_abbr"].to_numpy(),
            "AWAY_TEAM_ID": events["away_abbr"].map(team_id).to_numpy(),
            "AWAY_ABBR": events["away_abbr"].to_numpy(),
        }
    )
    players = (
        ok.drop_duplicates("PLAYER_ID")[["event_id", "PLAYER_ID", "player", "TEAM_ID", "TEAM_ABBREVIATION"]]
        .rename(columns={"event_id": "GAME_ID", "player": "PLAYER_NAME"})
    )

    features = build_features(hp, ht, games, players)
    up = features[features["IS_UPCOMING"]].copy()

    validate_serving_frame(up, models.features["minutes"])
    up["MIN_PRED"] = np.clip(models.minutes.predict(up[models.features["minutes"]].astype(float)), 0, 60)
    props = sorted({MARKET_TO_PROP[m] for m in ok["market"]})
    for prop in props:
        validate_serving_frame(up, models.features[prop])
        up[f"proj_{prop}"] = np.clip(models.props[prop].predict(up[models.features[prop]].astype(float)), 0, 80)

    out = ok.merge(up[["PLAYER_ID", "MIN_PRED"] + [f"proj_{p}" for p in props]], on="PLAYER_ID", how="left")
    out["projection"] = np.nan
    for market, prop in MARKET_TO_PROP.items():
        if prop in props:
            mask = out["market"] == market
            out.loc[mask, "projection"] = out.loc[mask, f"proj_{prop}"]

    out["opponent"] = np.where(out["TEAM_ABBREVIATION"] == out["home_abbr"], out["away_abbr"], out["home_abbr"])
    out["difference"] = out["projection"] - out["line"]
    out["mins_before_tip"] = (out["commence_time"] - out["captured_at"]).dt.total_seconds() / 60
    out["_gap"] = out["difference"].abs()
    slate = (
        out.rename(columns={"home_abbr": "home", "away_abbr": "away", "PLAYER_ID": "player_id",
                            "TEAM_ABBREVIATION": "team", "MIN_PRED": "min_pred"})
        .sort_values(["tip_et", "_gap"], ascending=[True, False])
        [SLATE_COLUMNS]
        .reset_index(drop=True)
    )
    return SlateResult(slate, skipped_df, meta)


MARKET_LABEL = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST"}
OUT_DIR = Path("data/processed")


def render_slate(result: SlateResult) -> str:
    """Plain-text slate: projection, line and difference per player, then who was left off and why."""
    meta, slate, skipped = result.meta, result.slate, result.skipped
    out = [
        f"Slate for {meta['slate_date']}  (history through {meta['history_last_date']}; "
        f"models {meta['model_label']}, trained through {meta['model_trained_through']})",
        f"{meta['n_projected']} projected of {meta['n_lines']} lines; {meta['n_skipped']} skipped.",
        "Projection is the model's number; Diff = Projection - Line. Not advice.",
        "",
    ]
    if slate.empty:
        out.append("No projections for this slate.")
    for (event, tip), game in slate.groupby(["event_id", "tip_et"], sort=False):
        first = game.iloc[0]
        out.append(f"{first['away']} @ {first['home']}   {tip.strftime('%-I:%M %p')} ET")
        out.append(f"  {'Player':<24}{'Team':<6}{'Mkt':<5}{'Line':>6}{'Projection':>12}{'Diff':>7}{'Min':>6}{'Books':>7}")
        for r in game.itertuples():
            out.append(
                f"  {r.player[:23]:<24}{r.team:<6}{MARKET_LABEL.get(r.market, r.market):<5}"
                f"{r.line:>6.1f}{r.projection:>12.1f}{r.difference:>+7.1f}{r.min_pred:>6.0f}{r.books:>7}"
            )
        out.append("")
    if len(skipped):
        out.append(f"Skipped ({len(skipped)}):")
        for reason, group in skipped.groupby("reason"):
            names = ", ".join(sorted(group["player"].unique())[:12])
            more = group["player"].nunique() - 12
            out.append(f"  {reason}: {len(group)}  ({names}{f', +{more} more' if more > 0 else ''})")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.common.env import load_env_file
    from src.common.seasons import season_label
    from src.features.history import load_history
    from src.serving.lines import load_line_log
    from src.serving.models import load_models
    from src.serving.rosters import fetch_roster

    ap = argparse.ArgumentParser(description="Print tonight's projections next to the posted lines.")
    ap.add_argument("--date", help="slate date, YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--include-started", action="store_true", help="keep games that already tipped off")
    ap.add_argument("--log-path", help="captured lines CSV (default: LINES_LOG_PATH from the environment or .env)")
    ap.add_argument("--out", help="CSV to write (default: data/processed/slate_<date>.csv)")
    args = ap.parse_args(argv)

    load_env_file()
    log_path = args.log_path or os.getenv("LINES_LOG_PATH")
    if not log_path:
        ap.error("no lines log: pass --log-path or set LINES_LOG_PATH in .env")

    now = pd.Timestamp.now(tz="UTC")
    slate_date = dt.date.fromisoformat(args.date) if args.date else now.tz_convert(EASTERN).date()
    season = season_for_date(slate_date)
    start = int(season[:4])
    seasons = [season_label(start - 1), season]

    print(f"Loading history for {seasons} (current season refreshed)...")
    players, teams = load_history(seasons, refresh_seasons=[season])
    result = build_slate(
        load_line_log(log_path), fetch_roster(season), players, teams, load_models(),
        as_of=now, slate_date=slate_date, include_started=args.include_started,
    )
    print(render_slate(result))

    out = Path(args.out) if args.out else OUT_DIR / f"slate_{slate_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.slate.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

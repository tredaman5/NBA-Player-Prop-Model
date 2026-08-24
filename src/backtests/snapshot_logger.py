"""
snapshot_logger.py

Appends each day's betslip results (model projection + sportsbook line +
edge/EV/decision, from slip_from_csv.py) to a persistent, season-long log
so real backtest metrics can be computed once games complete.

Unlike every other file in data/processed/, this log is never overwritten
wholesale -- each day's rows get appended (or replaced, if the same
log_date is re-run), so it accumulates across the season.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_LOG_PATH = DATA_PROCESSED / "prediction_log.csv"

BACKFILL_COLUMNS = ["actual_value", "actual_result", "model_error", "line_error"]
ACTUAL_STAT_COL = {"points": "PTS", "rebounds": "REB", "assists": "AST"}


def append_daily_snapshot(
    betslip_results_path: Path,
    log_date: str,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Reads a day's betslip_results.csv (from slip_from_csv.run_slip) and
    appends it to the persistent prediction log, tagged with log_date.

    If log_date already has rows in the log (e.g. re-run same day), those
    rows are replaced rather than duplicated. Rows with a non-empty
    "error" column (no matching prediction found) are dropped -- nothing
    to backtest there.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    day = pd.read_csv(betslip_results_path)
    if "error" in day.columns:
        day = day[day["error"].fillna("") == ""].copy()
    day = day.drop(columns=[c for c in ["error"] if c in day.columns])

    day["log_date"] = log_date
    for c in ["actual_value", "model_error", "line_error"]:
        day[c] = np.nan
    day["actual_result"] = ""

    if log_path.exists():
        existing = pd.read_csv(log_path)
        existing = existing[existing["log_date"] != log_date]
        combined = pd.concat([existing, day], ignore_index=True)
    else:
        combined = day

    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(log_path, index=False)
    return combined


def backfill_results(
    log_path: Optional[Path] = None,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fills in actual_value/actual_result/model_error/line_error for logged
    rows whose games have completed, using the same *_with_pred parquet
    files the rest of the pipeline already produces (no new API calls).

    Rows for games that haven't been built into a *_with_pred file yet are
    left pending -- safe to re-run daily as more games complete.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    if processed_dir is None:
        processed_dir = DATA_PROCESSED
    if not log_path.exists():
        raise FileNotFoundError(f"No prediction log found at {log_path}")

    log = pd.read_csv(log_path)
    log["game_date"] = pd.to_datetime(log["game_date"])
    # An all-empty-string column round-trips through CSV as float64 NaN;
    # force it back to string so later WIN/LOSS/PUSH assignments don't warn.
    log["actual_result"] = log["actual_result"].fillna("").astype(str)

    pending = log[log["actual_result"] == ""]
    if pending.empty:
        return log

    st = season_type.replace(" ", "")

    for prop, col in ACTUAL_STAT_COL.items():
        prop_rows = pending[pending["prop"] == prop]
        if prop_rows.empty:
            continue

        results_path = processed_dir / f"{prop}_with_pred_{season}_{st}.parquet"
        if not results_path.exists():
            continue  # not built yet; leave these rows pending

        actuals = pd.read_parquet(results_path, columns=["PLAYER_NAME", "GAME_DATE", col])
        actuals["GAME_DATE"] = pd.to_datetime(actuals["GAME_DATE"])
        actuals["_key_name"] = actuals["PLAYER_NAME"].astype(str).str.strip().str.lower()

        for idx, row in prop_rows.iterrows():
            key_name = str(row["player_name"]).strip().lower()
            match = actuals[(actuals["_key_name"] == key_name) & (actuals["GAME_DATE"] == row["game_date"])]
            if match.empty:
                continue

            actual_value = float(match.iloc[0][col])
            line = float(row["line"])
            chosen_side = str(row["chosen_side"])

            if actual_value > line:
                outcome = "over"
            elif actual_value < line:
                outcome = "under"
            else:
                outcome = "push"

            if outcome == "push":
                result = "PUSH"
            elif outcome == chosen_side:
                result = "WIN"
            else:
                result = "LOSS"

            log.loc[idx, "actual_value"] = actual_value
            log.loc[idx, "actual_result"] = result
            log.loc[idx, "model_error"] = actual_value - float(row["mu_pred"])
            log.loc[idx, "line_error"] = actual_value - line

    log.to_csv(log_path, index=False)
    return log

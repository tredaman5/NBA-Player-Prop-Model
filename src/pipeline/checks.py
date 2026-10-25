"""
checks.py: pass/fail rules around the slate build.

build_slate already refuses a stale history and a roster missing a team. These are the rules that
need the whole run's numbers: are there games to project, did the names still match, how old are the
models. A failed rule raises; the daily command turns that into a non-zero exit.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from src.common.validation import DataQualityError
from src.serving.slate import EASTERN, SlateResult

UNMATCHED_REASONS = ("unknown team name", "not on either team's roster", "ambiguous name")
MAX_UNMATCHED_SHARE = 0.2
MAX_MODEL_AGE_DAYS = 60


class NoLinesError(DataQualityError):
    """There is nothing to project for the date: no games in the log, or every game has started."""


def check_lines_for_date(log: pd.DataFrame, slate_date: dt.date) -> None:
    tips = log["commence_time"].dt.tz_convert(EASTERN).dt.date
    if not (tips == slate_date).any():
        raise NoLinesError(
            f"the lines log has no games on {slate_date} (no games that day, or the capture missed them)"
        )


def check_result(result: SlateResult, max_unmatched_share: float = MAX_UNMATCHED_SHARE) -> None:
    meta = result.meta
    n_lines = meta["n_lines"]
    if n_lines == 0:
        raise NoLinesError("no lines left to project: every captured game for that date has already started")
    unmatched = int(result.skipped["reason"].isin(UNMATCHED_REASONS).sum())
    share = unmatched / n_lines
    if share > max_unmatched_share:
        raise DataQualityError(
            f"{unmatched} of {n_lines} lines ({share:.0%}) could not be matched to a team or player; "
            "a provider name or roster change may have broken the mapping"
        )
    if meta["n_projected"] == 0:
        raise DataQualityError(f"0 of {n_lines} lines could be projected; see the skipped list for why")


def model_age_warning(meta: dict, slate_date: dt.date, max_days: int = MAX_MODEL_AGE_DAYS) -> Optional[str]:
    trained = pd.Timestamp(meta["model_trained_through"]).date()
    age = (slate_date - trained).days
    if age > max_days:
        return f"models were trained through {trained}, {age} days before this slate; retrain"
    return None

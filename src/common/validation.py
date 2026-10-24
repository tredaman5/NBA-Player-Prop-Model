"""
validation.py

Data-quality gates. A source that silently degrades (empty results, a changed
schema, duplicates, impossible values) must stop the pipeline loudly instead of
letting it quietly train on junk. When the 2025-26 inactive-player pull returned
zero rows for the whole season (a third of its calls also failed), the build
printed a warning and trained on the empty feature anyway.

Each validator collects every problem it finds and raises one DataQualityError
listing all of them, so a bad pull is diagnosed in one run.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


class DataQualityError(ValueError):
    """Raised when data fails a quality gate. The pipeline should stop, not continue."""


class EmptyDataError(DataQualityError):
    """A source returned nothing. Fatal for a finished season; expected for one that has not started."""


PLAYER_LOG_COLUMNS = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "MATCHUP", "MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV",
]
TEAM_LOG_COLUMNS = [
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
    "PTS", "REB", "AST", "FGA", "FTA", "TOV",
]

MAX_MINUTES = 70  # a four-overtime game is about 68 minutes
COUNTING_STATS = ["PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"]


def _raise_if(problems: List[str], source: str) -> None:
    if problems:
        raise DataQualityError(f"{source} failed data-quality checks:\n  - " + "\n  - ".join(problems))


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataQualityError(f"{source}: missing columns {missing}; the provider's schema may have changed")


def validate_player_gamelog(df: pd.DataFrame, source: str = "player game log") -> None:
    if df.empty:
        raise EmptyDataError(f"{source} is empty")
    _require_columns(df, PLAYER_LOG_COLUMNS, source)

    problems: List[str] = []

    for col in ["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "GAME_DATE"]:
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"{n} rows have a missing {col}")

    n = int(df.duplicated(["GAME_ID", "PLAYER_ID"]).sum())
    if n:
        problems.append(f"{n} duplicate (GAME_ID, PLAYER_ID) rows")

    minutes = pd.to_numeric(df["MIN"], errors="coerce")
    n = int(((minutes < 0) | (minutes > MAX_MINUTES)).sum())
    if n:
        problems.append(f"{n} rows have impossible MIN (outside 0-{MAX_MINUTES})")

    for col in COUNTING_STATS:
        n = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
        if n:
            problems.append(f"{n} rows have a negative {col}")

    _raise_if(problems, source)


def validate_team_gamelog(df: pd.DataFrame, source: str = "team game log") -> None:
    if df.empty:
        raise EmptyDataError(f"{source} is empty")
    _require_columns(df, TEAM_LOG_COLUMNS, source)

    problems: List[str] = []

    sides = df.groupby("GAME_ID")["TEAM_ABBREVIATION"].nunique()
    rows = df.groupby("GAME_ID").size()
    bad = sides[(sides != 2) | (rows != 2)]
    if len(bad):
        problems.append(f"{len(bad)} games do not have exactly 2 team rows (e.g. {list(bad.index[:3])})")

    for col in ["PTS", "FGA", "FTA", "TOV"]:
        n = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
        if n:
            problems.append(f"{n} rows have a negative {col}")

    _raise_if(problems, source)


def validate_training_frame(df: pd.DataFrame, features: Iterable[str], max_null_share: float = 0.2) -> None:
    """
    Gate before training: every feature must exist, vary, and be mostly present.

    A constant column is what a source that silently went empty looks like; a mostly
    missing column is what a partial pull looks like. (Rolling features are legitimately
    null for each player's first game, so the null limit is generous.)
    """
    features = list(features)
    problems: List[str] = []

    for col in features:
        if col not in df.columns:
            problems.append(f"feature {col} is not in the dataset")
            continue
        null_share = float(df[col].isna().mean())
        if null_share > max_null_share:
            problems.append(f"feature {col} is {null_share:.0%} missing (limit {max_null_share:.0%})")
        elif df[col].nunique(dropna=True) <= 1:
            problems.append(f"feature {col} is constant, so the source behind it may have returned nothing")

    _raise_if(problems, "training data")


def validate_serving_frame(df: pd.DataFrame, features: Iterable[str], max_null_share: float = 0.5) -> None:
    """
    Gate before predicting for tonight: every feature must exist and be mostly present.

    Unlike validate_training_frame, a constant column is fine: a one-game slate legitimately has
    them. A feature that is mostly null means the features were built from a broken history.
    """
    problems: List[str] = []
    for col in features:
        if col not in df.columns:
            problems.append(f"feature {col} is not in the dataset")
            continue
        share = float(df[col].isna().mean()) if len(df) else 0.0
        if share > max_null_share:
            problems.append(f"feature {col} is {share:.0%} missing (limit {max_null_share:.0%})")
    _raise_if(problems, "serving data")

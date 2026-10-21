"""NBA season naming helpers."""

from __future__ import annotations

import pandas as pd


def season_label(start_year: int) -> str:
    """2025 -> "2025-26"."""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def season_for_date(date) -> str:
    """The season a game date belongs to. Games run October to June, so September on starts a new one."""
    d = pd.Timestamp(date)
    return season_label(d.year if d.month >= 9 else d.year - 1)

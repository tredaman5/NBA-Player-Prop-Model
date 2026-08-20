"""
build_dataset.py (rebounds props)

Builds a player-game dataset for training a REBOUNDS model.
Uses minutes predictions + rolling reb opportunity features.

Output:
- data/processed/rebounds_dataset_<season>_<season_type>.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.opponent_stats import (
    build_opponent_defense_table,
    fetch_league_team_gamelogs,
    merge_opponent_features,
    opponent_feature_cols,
)
from src.common.usage_stats import add_usage_rate_rolling, usage_rate_feature_cols


REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def _default_minutes_pred_path(season: str, season_type: str) -> Path:
    return DATA_PROCESSED / f"minutes_with_pred_{season}_{season_type.replace(' ', '')}.parquet"


def _add_rolling_reb_opportunity(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    out = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    min_safe = out["MIN"].replace(0, np.nan)
    out["REB_PER_MIN"] = out["REB"] / min_safe
    out["REB_PER_MIN"] = out["REB_PER_MIN"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in windows:
        out[f"REB_PER_MIN_ROLL_{w}"] = (
            out.groupby("PLAYER_ID")["REB_PER_MIN"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return out


def build_rebounds_dataset(
    season: str,
    season_type: str = "Regular Season",
    minutes_with_pred_path: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    if minutes_with_pred_path is None:
        minutes_with_pred_path = _default_minutes_pred_path(season, season_type)

    if not minutes_with_pred_path.exists():
        raise FileNotFoundError(
            f"Minutes-with-pred file not found: {minutes_with_pred_path}\n"
            f"Run: python -m src.minutes.predict"
        )

    df = pd.read_parquet(minutes_with_pred_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    needed = ["REB", "MIN", "MIN_PRED"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in minutes-with-pred dataset.")

    df["REB"] = pd.to_numeric(df["REB"], errors="coerce")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.dropna(subset=["REB", "MIN"]).copy()

    df = _add_rolling_reb_opportunity(df)

    # Opponent defense context (how many rebounds this opponent has been
    # allowing recently, shifted so no future info leaks in)
    team_gamelog = fetch_league_team_gamelogs(season, season_type=season_type, use_cache=use_cache)
    defense_table = build_opponent_defense_table(team_gamelog, stat_cols=("REB",))
    opp_cols = opponent_feature_cols("REB")
    df = merge_opponent_features(df, defense_table, opp_cols)

    # Usage rate: rolling share of the team's possessions-used this player
    # accounted for (pace-normalized, unlike the per-minute rates above)
    df = add_usage_rate_rolling(df, team_gamelog)
    usage_cols = usage_rate_feature_cols()

    feature_cols = [
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "MIN_ROLL_3",
        "MIN_ROLL_5",
        "MIN_ROLL_10",
        "MIN_PRED",
        "REB_PER_MIN_ROLL_3",
        "REB_PER_MIN_ROLL_5",
        "REB_PER_MIN_ROLL_10",
        "TEAM_PTS_ROLL_5",
        "TEAM_PTS_ROLL_10",
        *opp_cols,
        *usage_cols,
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPP_ABBREVIATION",
        "REB",  # target
        "MIN",  # actual minutes
    ] + feature_cols

    out = df[keep_cols].copy()

    out_path = DATA_PROCESSED / f"rebounds_dataset_{season}_{season_type.replace(' ', '')}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved rebounds dataset -> {out_path}")
    print(f"Rows: {len(out):,} | Features: {len(feature_cols)}")

    return out


if __name__ == "__main__":
    build_rebounds_dataset(season="2025-26", season_type="Regular Season")

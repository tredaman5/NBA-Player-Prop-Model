"""
walk_forward.py

Reads the accumulated, backfilled prediction_log.csv (built by
snapshot_logger.py across the season) and reports real backtest metrics:

- Model vs market: whose point estimate is closer to the actual outcome,
  the model's projection or the sportsbook line -- the direct test of
  whether the model beats the market, independent of any bet decision.
- Bet performance: hit rate on BET decisions, overall and by edge bucket
  (checks whether higher edge actually means a better hit rate -- if not,
  the edge number isn't calibrated to real outcomes yet).

Everything before this file only checked the model against naive
rolling-average baselines. This is the real test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_LOG_PATH = DATA_PROCESSED / "prediction_log.csv"


def load_backfilled_log(log_path: Optional[Path] = None) -> pd.DataFrame:
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log found at {log_path}\n"
            f"Run snapshot_logger.append_daily_snapshot() during the season first."
        )
    log = pd.read_csv(log_path)
    return log[log["actual_result"].fillna("") != ""].copy()


def model_vs_market_report(log: pd.DataFrame) -> pd.DataFrame:
    """
    For each prop, compares MAE(model_projection, actual) vs
    MAE(sportsbook_line, actual). A positive model_beats_line_by means the
    model's point estimate is genuinely closer to reality than the market's.
    """
    rows = []
    for prop, g in log.groupby("prop"):
        model_mae = (g["mu_pred"] - g["actual_value"]).abs().mean()
        line_mae = (g["line"] - g["actual_value"]).abs().mean()
        rows.append(
            {
                "prop": prop,
                "n_games": len(g),
                "model_mae": model_mae,
                "line_mae": line_mae,
                "model_beats_line_by": line_mae - model_mae,
            }
        )
    return pd.DataFrame(rows)


def bet_performance_report(log: pd.DataFrame, edge_buckets: Iterable[float] = (0.02, 0.05, 0.10)) -> pd.DataFrame:
    """
    Hit rate on BET decisions, overall and by edge bucket.
    """
    bets = log[log["decision"] == "BET"].copy()
    if bets.empty:
        return pd.DataFrame(columns=["bucket", "n_bets", "hit_rate"])

    thresholds = sorted(edge_buckets)

    def _bucket(edge: float) -> str:
        for b in thresholds:
            if edge < b:
                return f"<{b:.0%}"
        return f">={thresholds[-1]:.0%}"

    bets["bucket"] = bets["edge"].astype(float).map(_bucket)
    bets["hit"] = (bets["actual_result"] == "WIN").astype(int)

    rows = [{"bucket": "ALL", "n_bets": len(bets), "hit_rate": bets["hit"].mean()}]
    for bucket, g in bets.groupby("bucket"):
        rows.append({"bucket": bucket, "n_bets": len(g), "hit_rate": g["hit"].mean()})
    return pd.DataFrame(rows)


def run_backtest_report(log_path: Optional[Path] = None) -> None:
    log = load_backfilled_log(log_path)
    if log.empty:
        print("No backfilled results yet -- nothing to report.")
        return

    print(f"\n=== Model vs Market ({len(log)} backfilled rows) ===")
    print(model_vs_market_report(log).to_string(index=False))

    print("\n=== Bet Performance ===")
    print(bet_performance_report(log).to_string(index=False))


if __name__ == "__main__":
    run_backtest_report()

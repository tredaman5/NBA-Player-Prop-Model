"""
Tests for src/backtests/snapshot_logger.py:
- appending accumulates across days instead of overwriting
- re-running the same log_date replaces that day's rows, not duplicates
- backfill correctly matches actuals by (player_name, game_date) and
  scores WIN/LOSS/PUSH against the chosen side
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtests.snapshot_logger import append_daily_snapshot, backfill_results


def _betslip_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "betslip_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_row(**overrides):
    row = {
        "prop": "points",
        "player_name": "Test Player",
        "game_date": "2026-01-01",
        "line": 20.5,
        "mu_pred": 22.0,
        "edge": 0.06,
        "decision": "BET",
        "chosen_side": "over",
        "error": "",
    }
    row.update(overrides)
    return row


def test_append_accumulates_across_different_days(tmp_path):
    log_path = tmp_path / "prediction_log.csv"

    day1 = _betslip_csv(tmp_path, [_base_row(game_date="2026-01-01")])
    append_daily_snapshot(day1, log_date="2026-01-01", log_path=log_path)

    day2_path = tmp_path / "betslip_results_day2.csv"
    pd.DataFrame([_base_row(game_date="2026-01-02")]).to_csv(day2_path, index=False)
    result = append_daily_snapshot(day2_path, log_date="2026-01-02", log_path=log_path)

    assert len(result) == 2
    assert set(result["log_date"]) == {"2026-01-01", "2026-01-02"}


def test_rerunning_same_log_date_replaces_not_duplicates(tmp_path):
    log_path = tmp_path / "prediction_log.csv"

    day1 = _betslip_csv(tmp_path, [_base_row(mu_pred=22.0)])
    append_daily_snapshot(day1, log_date="2026-01-01", log_path=log_path)

    day1_rerun = _betslip_csv(tmp_path, [_base_row(mu_pred=23.0)])
    result = append_daily_snapshot(day1_rerun, log_date="2026-01-01", log_path=log_path)

    assert len(result) == 1
    assert result.iloc[0]["mu_pred"] == 23.0


def test_error_rows_are_dropped(tmp_path):
    path = _betslip_csv(
        tmp_path, [_base_row(), _base_row(player_name="No Match", error="No matching row")]
    )
    result = append_daily_snapshot(path, log_date="2026-01-01", log_path=tmp_path / "log.csv")
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Test Player"


def test_backfill_scores_win_loss_push_correctly(tmp_path):
    log_path = tmp_path / "prediction_log.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    rows = [
        _base_row(player_name="Winner", chosen_side="over", line=20.5, mu_pred=22.0),  # actual 25 -> WIN
        _base_row(player_name="Loser", chosen_side="over", line=20.5, mu_pred=22.0),   # actual 10 -> LOSS
        _base_row(player_name="Pusher", chosen_side="over", line=20.0, mu_pred=22.0),  # actual 20 -> PUSH
    ]
    betslip = _betslip_csv(tmp_path, rows)
    append_daily_snapshot(betslip, log_date="2026-01-01", log_path=log_path)

    actuals = pd.DataFrame(
        [
            {"PLAYER_NAME": "Winner", "GAME_DATE": "2026-01-01", "PTS": 25},
            {"PLAYER_NAME": "Loser", "GAME_DATE": "2026-01-01", "PTS": 10},
            {"PLAYER_NAME": "Pusher", "GAME_DATE": "2026-01-01", "PTS": 20},
        ]
    )
    actuals.to_parquet(processed_dir / "points_with_pred_2025-26_RegularSeason.parquet", index=False)

    result = backfill_results(
        log_path=log_path, season="2025-26", season_type="Regular Season", processed_dir=processed_dir
    )
    result = result.set_index("player_name")

    assert result.loc["Winner", "actual_result"] == "WIN"
    assert result.loc["Loser", "actual_result"] == "LOSS"
    assert result.loc["Pusher", "actual_result"] == "PUSH"
    assert result.loc["Winner", "actual_value"] == 25
    assert result.loc["Winner", "model_error"] == 25 - 22.0
    assert result.loc["Winner", "line_error"] == 25 - 20.5


def test_backfill_leaves_unmatched_rows_pending(tmp_path):
    log_path = tmp_path / "prediction_log.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    betslip = _betslip_csv(tmp_path, [_base_row(player_name="Nobody Yet")])
    append_daily_snapshot(betslip, log_date="2026-01-01", log_path=log_path)

    # No *_with_pred parquet exists yet -- game hasn't been built/played.
    result = backfill_results(
        log_path=log_path, season="2025-26", season_type="Regular Season", processed_dir=processed_dir
    )
    assert result.iloc[0]["actual_result"] == "" or pd.isna(result.iloc[0]["actual_result"])

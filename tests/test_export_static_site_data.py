"""
Tests for src/ui/export_static_site_data.py:
- lift_pct is computed correctly from mae/baseline_mae
- missing eval files are skipped, not errors
- missing betslip_results.csv gives todays_slate=None (not an error)
"""

from __future__ import annotations

import json

import pandas as pd

from src.ui.export_static_site_data import export_site_data


def _write_eval_json(artifacts_dir, prop, mae, baseline_mae, season="2024-25", season_type="RegularSeason"):
    payload = {
        "mae": mae,
        "rmse": mae + 1.0,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_mae + 1.0,
        "rows_test": 100,
        "test_date_min": "2025-01-01",
        "test_date_max": "2025-02-01",
    }
    path = artifacts_dir / f"{prop}_eval_{season}_{season_type}.json"
    path.write_text(json.dumps(payload))


def test_lift_pct_computed_correctly(tmp_path):
    artifacts_dir = tmp_path / "models_artifacts"
    artifacts_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    _write_eval_json(artifacts_dir, "points", mae=4.77, baseline_mae=4.818)

    data = export_site_data(
        artifacts_dir=artifacts_dir,
        processed_dir=processed_dir,
        season="2024-25",
        season_type="Regular Season",
        out_path=tmp_path / "data.json",
    )

    points_row = next(r for r in data["model_performance"] if r["prop"] == "points")
    expected_lift = round((4.818 - 4.77) / 4.818 * 100, 2)
    assert points_row["lift_pct"] == expected_lift


def test_missing_eval_files_are_skipped_not_errors(tmp_path):
    artifacts_dir = tmp_path / "models_artifacts"
    artifacts_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Only points exists; rebounds/assists eval files are missing.
    _write_eval_json(artifacts_dir, "points", mae=4.77, baseline_mae=4.818)

    data = export_site_data(
        artifacts_dir=artifacts_dir,
        processed_dir=processed_dir,
        season="2024-25",
        season_type="Regular Season",
        out_path=tmp_path / "data.json",
    )

    assert len(data["model_performance"]) == 1
    assert data["model_performance"][0]["prop"] == "points"


def test_missing_betslip_results_gives_none_not_error(tmp_path):
    artifacts_dir = tmp_path / "models_artifacts"
    artifacts_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    data = export_site_data(
        artifacts_dir=artifacts_dir,
        processed_dir=processed_dir,
        out_path=tmp_path / "data.json",
    )
    assert data["todays_slate"] is None


def test_betslip_error_rows_are_excluded_from_slate(tmp_path):
    artifacts_dir = tmp_path / "models_artifacts"
    artifacts_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    pd.DataFrame(
        [
            {"game_date": "2026-01-01", "player_name": "Good Row", "prop": "points", "chosen_side": "over",
             "line": 20.5, "mu_pred": 22.0, "edge": 0.06, "ev_per_dollar": 0.1, "decision": "BET", "error": ""},
            {"game_date": "2026-01-01", "player_name": "Bad Row", "prop": "points", "chosen_side": "",
             "line": "", "mu_pred": "", "edge": "", "ev_per_dollar": "", "decision": "", "error": "no match"},
        ]
    ).to_csv(processed_dir / "betslip_results.csv", index=False)

    data = export_site_data(
        artifacts_dir=artifacts_dir,
        processed_dir=processed_dir,
        out_path=tmp_path / "data.json",
    )
    assert len(data["todays_slate"]) == 1
    assert data["todays_slate"][0]["player_name"] == "Good Row"

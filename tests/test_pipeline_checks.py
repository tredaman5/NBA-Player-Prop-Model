import datetime as dt

import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.pipeline.checks import (
    NoLinesError, check_lines_for_date, check_result, model_age_warning,
)
from src.serving.slate import SlateResult
from tests.slate_fixture import SLATE_DATE, make_log


def _result(n_lines, n_projected, reasons=()):
    skipped = pd.DataFrame({
        "player": [f"p{i}" for i in range(len(reasons))], "event_id": "e1",
        "market": "player_points", "reason": list(reasons),
    })
    meta = {"n_lines": n_lines, "n_projected": n_projected, "n_skipped": len(reasons),
            "model_trained_through": "2025-01-01"}
    return SlateResult(pd.DataFrame(), skipped, meta)


def test_a_log_with_a_game_on_the_date_passes():
    check_lines_for_date(make_log(), SLATE_DATE)


def test_a_log_with_no_game_on_the_date_raises_no_lines():
    with pytest.raises(NoLinesError, match="no games"):
        check_lines_for_date(make_log(), dt.date(2025, 1, 26))


def test_no_lines_is_still_a_data_quality_error():
    assert issubclass(NoLinesError, DataQualityError)


def test_a_healthy_result_passes_even_with_thin_and_rookie_skips():
    check_result(_result(100, 85, ["line from only 2 book(s)"] * 8 + ["only 1 prior games"] * 7))


def test_many_unmatched_names_fail_the_run():
    with pytest.raises(DataQualityError, match="matched"):
        check_result(_result(100, 60, ["not on either team's roster"] * 40))


def test_unknown_team_names_count_as_unmatched():
    with pytest.raises(DataQualityError, match="matched"):
        check_result(_result(10, 5, ["unknown team name"] * 5))


def test_nothing_projected_fails_the_run():
    with pytest.raises(DataQualityError, match="0 of 5"):
        check_result(_result(5, 0, ["only 1 prior games"] * 5))


def test_no_lines_left_raises_no_lines():
    with pytest.raises(NoLinesError, match="started"):
        check_result(_result(0, 0))


def test_old_models_give_a_warning_and_fresh_ones_do_not():
    old = {"model_trained_through": "2024-10-01"}
    assert "retrain" in model_age_warning(old, SLATE_DATE)
    assert model_age_warning({"model_trained_through": "2025-01-01"}, SLATE_DATE) is None

import numpy as np
import pandas as pd
import pytest

from src.backtests.metrics import summarize
from src.backtests.rules import BREAKEVEN, VERDICT_MIN_LINES

COLS = ["event_id", "game_date", "player", "market", "line", "books", "mins_before_tip", "be_over", "be_under",
        "projection", "actual", "status", "model_label", "min_pred", "prior_games"]


def _frame(rows, label="m"):
    df = pd.DataFrame(rows, columns=["event_id", "line", "projection", "actual"])
    df["game_date"] = pd.Timestamp("2026-11-01").date()
    df["player"] = [f"p{i}" for i in range(len(df))]
    df["market"] = "player_points"
    df["books"] = 6
    df["mins_before_tip"] = 120.0
    df["be_over"] = 0.52
    df["be_under"] = 0.52
    df["status"] = "scored"
    df["model_label"] = label
    return df


def _synthetic(n, hit_every=1, events=300, project=2.0, actual_over=3.0):
    """n lines with line 10.5, projection line+project; every `hit_every`-th line lands over, the rest under."""
    rows = []
    for i in range(n):
        hit = (i % hit_every == 0) if hit_every > 1 else True
        rows.append((f"e{i % events}", 10.5, 10.5 + project, 10.5 + actual_over if hit else 10.5 - actual_over))
    return _frame(rows)


def test_hand_checked_numbers():
    df = _frame([
        ("e1", 10.5, 12.0, 15),   # over, hit,  d = 4.5 - 3.0 = +1.5
        ("e1", 10.5, 9.0, 12),    # under, miss, d = 1.5 - 3.0 = -1.5
        ("e2", 20.0, 22.0, 20),   # push: dropped
        ("e2", 8.5, 8.5, 12),     # projection equals the line: dropped
        ("e2", 5.5, 7.0, 9),      # over, hit,  d = 3.5 - 2.0 = +1.5
    ])
    s = summarize(df)
    assert (s["n_scored"], s["n_directional"]) == (5, 3)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["mean_d"] == pytest.approx(0.5)
    assert s["mae_model"] == pytest.approx((3 + 3 + 2) / 3)
    assert s["mae_market"] == pytest.approx((4.5 + 1.5 + 3.5) / 3)
    assert s["model_closer_share"] == pytest.approx(2 / 3)
    assert s["breakeven_mean"] == pytest.approx(0.52)
    assert s["verdict"] == "interim" and s["lines_needed"] == VERDICT_MIN_LINES - 3


def test_unscored_lines_are_counted_but_not_measured():
    df = _frame([("e1", 10.5, 12.0, 15)])
    extra = df.copy()
    extra["status"] = "did not play"
    extra[["projection", "actual"]] = np.nan
    s = summarize(pd.concat([df, extra], ignore_index=True))
    assert s["n_scored"] == 1 and s["status_counts"] == {"scored": 1, "did not play": 1}


def test_an_empty_or_unscored_frame_is_interim_with_undefined_rates():
    s = summarize(_frame([("e1", 10.5, 12.0, 15)]).iloc[0:0])
    assert s["n_scored"] == 0 and s["n_directional"] == 0
    assert s["hit_rate"] is None and s["hit_ci"] is None and s["verdict"] == "interim"


def test_the_verdict_waits_for_the_pre_registered_number_of_lines():
    assert summarize(_synthetic(VERDICT_MIN_LINES - 1))["verdict"] == "interim"
    s = summarize(_synthetic(VERDICT_MIN_LINES))
    assert s["verdict"] == "edge_shown" and s["hit_ci"][0] > BREAKEVEN and s["d_ci"][0] > 0 and s["lines_needed"] == 0


def test_a_coin_flip_model_shows_no_edge():
    s = summarize(_synthetic(VERDICT_MIN_LINES, hit_every=2))
    assert s["verdict"] == "no_edge_shown"


def test_both_conditions_are_required():
    # always on the right side, but the projection overshoots so far the line was the closer number
    rows = [(f"e{i % 300}", 10.5, 20.5, 11.5) for i in range(VERDICT_MIN_LINES)]
    s = summarize(_frame(rows))
    assert s["hit_ci"][0] > BREAKEVEN and s["d_ci"][0] < 0
    assert s["verdict"] == "no_edge_shown"


def test_the_bootstrap_is_reproducible_and_brackets_the_estimate():
    df = _synthetic(600, hit_every=3, events=120)
    a, b = summarize(df), summarize(df)
    assert a["hit_ci"] == b["hit_ci"] and a["d_ci"] == b["d_ci"]
    assert a["hit_ci"][0] <= a["hit_rate"] <= a["hit_ci"][1]
    assert a["d_ci"][0] <= a["mean_d"] <= a["d_ci"][1]


def test_buckets_partition_the_directional_lines_by_gap_size():
    df = _frame([("e1", 10.0, 10.5, 12), ("e1", 10.0, 11.5, 12), ("e2", 10.0, 12.5, 9), ("e2", 10.0, 14.0, 15)])
    sizes = [b["n"] for b in summarize(df)["buckets"]]
    assert sizes == [1, 1, 1, 1]
    assert sum(sizes) == summarize(df)["n_directional"]


def test_mixing_model_labels_is_refused_because_samples_are_never_pooled():
    df = pd.concat([_frame([("e1", 10.5, 12.0, 15)], "a"), _frame([("e2", 10.5, 12.0, 15)], "b")], ignore_index=True)
    with pytest.raises(ValueError, match="one model"):
        summarize(df)

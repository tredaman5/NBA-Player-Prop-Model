"""
Tests for src/odds/capture_lines.py. All HTTP is faked -- nothing here touches
the network or spends API credits.

Covers: over/under pairing into rows, the tip-off window, skipping recently
captured games, the credit budget guards, dry-run spending nothing, and the
API key never appearing in an error.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.odds.capture_lines import LOG_COLUMNS, OddsApiError, _require_api_key, capture_lines

NOW = datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)
KEY = "SECRET-KEY-123"


def _event(event_id: str, minutes_from_now: int) -> dict:
    commence = NOW + timedelta(minutes=minutes_from_now)
    return {
        "id": event_id,
        "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": "Home Team",
        "away_team": "Away Team",
    }


def _odds_payload() -> dict:
    def outcome(side, price, player="Player A", point=24.5):
        return {"name": side, "description": player, "price": price, "point": point}

    return {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [{"key": "player_points", "outcomes": [outcome("Over", -115), outcome("Under", -105)]}],
            },
            {
                "key": "fanduel",
                # Only an Over is posted here -- row should still be kept.
                "markets": [{"key": "player_points", "outcomes": [outcome("Over", -110, point=25.5)]}],
            },
        ]
    }


class FakeApi:
    """Stands in for the HTTP layer. Records calls; headers mimic quota usage."""

    def __init__(self, events, remaining=500, per_call=3, send_headers=True):
        self.events = events
        self.remaining = remaining
        self.per_call = per_call
        self.send_headers = send_headers
        self.odds_calls = []

    def __call__(self, url, params):
        assert params["apiKey"] == KEY
        if url.endswith("/events"):
            headers = {"x-requests-remaining": str(self.remaining)} if self.send_headers else {}
            return self.events, headers
        self.odds_calls.append(url.split("/events/")[1].split("/odds")[0])
        self.remaining -= self.per_call
        headers = (
            {"x-requests-remaining": str(self.remaining), "x-requests-last": str(self.per_call)}
            if self.send_headers
            else {}
        )
        return _odds_payload(), headers


def _run(tmp_path, api, **kwargs):
    kwargs.setdefault("sleep_sec", 0)
    return capture_lines(api_key=KEY, log_path=tmp_path / "log.csv", http_get=api, now=NOW, **kwargs)


def test_writes_one_row_per_book_market_player_line_with_over_under_paired(tmp_path):
    api = FakeApi([_event("e1", 30)])
    summary = _run(tmp_path, api)

    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log.columns) == LOG_COLUMNS
    assert summary["rows_written"] == 2

    dk = log[log["bookmaker"] == "draftkings"].iloc[0]
    assert (dk["over_price"], dk["under_price"], dk["line"]) == (-115, -105, 24.5)

    fd = log[log["bookmaker"] == "fanduel"].iloc[0]
    assert fd["over_price"] == -110 and pd.isna(fd["under_price"])


def test_only_captures_games_inside_the_tipoff_window(tmp_path):
    events = [
        _event("started", -10),   # already tipped off
        _event("soon", 30),
        _event("later", 90),
        _event("tomorrow", 600),  # outside the window
    ]
    api = FakeApi(events)
    summary = _run(tmp_path, api, within_minutes=120)

    assert api.odds_calls == ["soon", "later"]
    assert summary["events_listed"] == 4
    assert summary["events_eligible"] == 2


def test_skips_games_captured_recently_unless_gap_is_zero(tmp_path):
    first = FakeApi([_event("e1", 30)])
    _run(tmp_path, first)

    again = FakeApi([_event("e1", 30)])
    _run(tmp_path, again)
    assert again.odds_calls == []  # already captured -> no credits spent

    forced = FakeApi([_event("e1", 30)])
    _run(tmp_path, forced, min_gap_minutes=0)
    assert forced.odds_calls == ["e1"]


def test_planned_budget_leaves_the_reserve_and_takes_earliest_games_first(tmp_path):
    events = [_event(f"g{i}", 10 + i) for i in range(8)]
    api = FakeApi(events, remaining=40, per_call=3)
    summary = _run(tmp_path, api, reserve=25, max_credits=1000)

    # Each call costs 3 against a floor of 25: 40 -> 37 -> ... stops before dipping under.
    assert api.remaining >= 25
    assert api.odds_calls == [f"g{i}" for i in range(len(api.odds_calls))]  # earliest first
    assert summary["stopped_reason"] == "reserve"
    assert 0 < summary["events_captured"] < 8


def test_max_credits_caps_spending_even_without_quota_headers(tmp_path):
    events = [_event(f"g{i}", 10 + i) for i in range(5)]
    api = FakeApi(events, send_headers=False)
    summary = _run(tmp_path, api, max_credits=6)  # 3 markets x 1 region = 3 credits a game

    assert summary["events_captured"] == 2
    assert summary["credits_spent"] == 6
    assert summary["stopped_reason"] == "max_credits"


def test_dry_run_spends_nothing_and_writes_nothing(tmp_path):
    api = FakeApi([_event("e1", 30), _event("e2", 60)])
    summary = _run(tmp_path, api, dry_run=True)

    assert api.odds_calls == []
    assert not (tmp_path / "log.csv").exists()
    assert summary["events_eligible"] == 2
    assert summary["estimated_cost"] == 6


def test_api_key_never_appears_in_an_error_or_its_traceback(tmp_path):
    def exploding_http(url, params):
        raise ConnectionError(f"Max retries exceeded with url: /v4/x?apiKey={params['apiKey']}")

    with pytest.raises(OddsApiError) as exc:
        capture_lines(api_key=KEY, log_path=tmp_path / "log.csv", http_get=exploding_http, now=NOW)

    assert KEY not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        _require_api_key()

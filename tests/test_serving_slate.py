"""
End-to-end tests of build_slate on a synthetic league: Boston (BOS, team id 100, players 1-2) and
Detroit (DET, team id 200, players 3-4) have played 12 games, last on 2025-01-23. Tonight is
2025-01-25, BOS vs DET at 6:30pm ET. The stand-in models are simple functions of the features, so
the tests can check exactly what the slate does with them.
"""

import datetime as dt

import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.serving.slate import build_slate
from tests.slate_fixture import (
    AS_OF, SLATE_DATE,
    make_history as _league, make_log as _log, make_models as _models, make_roster as _roster,
)


def _build(log=None, roster=None, history=None, **kw):
    hp, ht = history or _league()
    kw.setdefault("as_of", AS_OF)
    kw.setdefault("slate_date", SLATE_DATE)
    return build_slate(log if log is not None else _log(), roster if roster is not None else _roster(), hp, ht, _models(), **kw)


def test_the_projection_is_the_models_output_on_the_builders_features():
    hp, _ = _league()
    last_five = hp[hp["PLAYER_ID"] == 1].sort_values("GAME_DATE")["MIN"].iloc[-5:].mean()
    row = _build().slate.query("player == 'P1'").iloc[0]
    assert row["min_pred"] == pytest.approx(last_five)      # minutes stub = MIN_ROLL_5, a mean of the last 5 games
    assert row["projection"] == pytest.approx(0.5 * last_five)  # points stub = 0.5 * MIN_PRED
    assert row["difference"] == pytest.approx(row["projection"] - row["line"])
    assert (row["team"], row["opponent"], row["player_id"]) == ("BOS", "DET", 1)


def test_the_slate_lists_every_player_with_a_good_line_and_sorts_by_gap():
    slate = _build().slate
    assert sorted(slate["player"]) == ["P1", "P2", "P3", "P4"]
    gaps = slate["difference"].abs().tolist()
    assert gaps == sorted(gaps, reverse=True)  # one game: sorted by gap within it


def test_a_player_with_too_little_history_is_skipped_with_the_number_of_games():
    hp, ht = _league()
    hp = hp[~((hp["PLAYER_ID"] == 4) & (hp["GAME_ID"].isin([f"g{i}" for i in range(1, 11)])))]  # P4: 2 games
    res = _build(history=(hp, ht))
    assert "P4" not in res.slate["player"].tolist()
    assert res.skipped.query("player == 'P4'").iloc[0]["reason"] == "only 2 prior games"


def test_an_unmatched_name_is_skipped_with_a_reason_not_dropped_silently():
    res = _build(log=_log(players=("P1", "Totally Unknown")))
    assert res.skipped.query("player == 'Totally Unknown'").iloc[0]["reason"] == "not on either team's roster"
    assert res.meta["n_skipped"] == 1 and res.meta["n_projected"] == 1


def test_a_thin_market_is_skipped_with_the_book_count():
    res = _build(log=_log(players=("P1",), books=("dk", "fd")))
    assert res.slate.empty
    assert res.skipped.iloc[0]["reason"] == "line from only 2 book(s)"


def test_an_unknown_team_name_skips_the_game_with_a_reason():
    res = _build(log=_log(home="Springfield Isotopes"))
    assert res.slate.empty and (res.skipped["reason"] == "unknown team name").all()


def test_a_game_that_has_started_is_excluded_unless_asked_for():
    late = pd.Timestamp("2025-01-26T00:30:00Z")
    assert _build(as_of=late).slate.empty
    assert len(_build(as_of=late, include_started=True).slate) == 4


def test_games_on_other_days_are_not_on_tonights_slate():
    tomorrow = _log(tip="2025-01-26T23:30:00+00:00")
    assert _build(log=tomorrow).slate.empty


def test_history_on_or_after_the_slate_date_is_ignored_so_a_replay_cannot_peek():
    hp, ht = _league()
    base = _build(history=(hp, ht)).slate.query("player == 'P1'").iloc[0]["projection"]
    # tonight's real box score sneaks into history under another game id: it must not matter
    leak = hp[hp["GAME_ID"] == "g12"].copy()
    leak["GAME_ID"], leak["GAME_DATE"], leak["MIN"] = "real-game-tonight", pd.Timestamp("2025-01-25"), 99
    leak_t = ht[ht["GAME_ID"] == "g12"].copy()
    leak_t["GAME_ID"], leak_t["GAME_DATE"] = "real-game-tonight", pd.Timestamp("2025-01-25")
    peeked = _build(history=(pd.concat([hp, leak]), pd.concat([ht, leak_t]))).slate.query("player == 'P1'").iloc[0]
    assert peeked["projection"] == pytest.approx(base)


def test_stale_history_inside_the_season_is_refused():
    with pytest.raises(DataQualityError, match="stale"):
        _build(slate_date=dt.date(2025, 2, 5), as_of=pd.Timestamp("2025-02-05T12:00:00Z"),
               log=_log(tip="2025-02-05T23:30:00+00:00"))


def test_a_gap_across_the_offseason_is_not_stale():
    log = _log(tip="2025-10-22T23:30:00+00:00")
    res = _build(slate_date=dt.date(2025, 10, 22), as_of=pd.Timestamp("2025-10-22T12:00:00Z"), log=log)
    assert len(res.slate) == 4  # opening week: last season's games are the history, and that is fine


def test_a_roster_missing_a_team_abbreviation_is_refused():
    roster = _roster()
    roster = roster[roster["TEAM_ABBREVIATION"] != "UTA"]
    with pytest.raises(DataQualityError, match="abbreviations"):
        _build(roster=roster)


def test_a_player_with_lines_in_two_markets_gets_a_projection_for_each():
    both = pd.concat([_log(players=("P1",)), _log(players=("P1",), market="player_rebounds", line=4.5)])
    slate = _build(log=both).slate
    assert sorted(slate["market"]) == ["player_points", "player_rebounds"]
    p, r = (slate.query(f"market == '{m}'").iloc[0]["projection"] for m in ("player_points", "player_rebounds"))
    assert p == pytest.approx(r * 0.5 / 0.2)  # the two stub models use different multiples of MIN_PRED


def test_the_meta_reports_what_the_model_and_history_were():
    meta = _build().meta
    assert meta["model_label"] == "test" and meta["model_trained_through"] == "2025-01-01"
    assert meta["history_last_date"] == "2025-01-23"
    assert meta["n_lines"] == 4


def test_render_shows_tip_time_player_line_projection_and_difference():
    from src.serving.slate import render_slate

    text = render_slate(_build())
    assert "6:30 PM ET" in text
    assert "P1" in text and "10.5" in text
    assert "Projection" in text and "Line" in text and "Diff" in text


def test_render_summarises_skipped_players_with_their_reasons():
    from src.serving.slate import render_slate

    text = render_slate(_build(log=_log(players=("P1", "Nobody Real"))))
    assert "Skipped" in text
    assert "Nobody Real" in text and "not on either team's roster" in text


def test_render_never_uses_betting_language():
    from src.serving.slate import render_slate

    text = render_slate(_build(log=_log(players=("P1", "Nobody Real")))).lower()
    for word in ("edge", "bet ", "ev ", "kelly", "probab", "pick"):
        assert word not in text


def test_render_handles_an_empty_slate():
    from src.serving.slate import render_slate

    text = render_slate(_build(log=_log(players=("Nobody Real",))))
    assert "No projections" in text

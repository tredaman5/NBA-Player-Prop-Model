"""
replay_lines on the synthetic league, extended to 13 games: game 13 is BOS (home) vs DET on 2025-01-25,
the night the log's lines are for. The box score for that night is in history, so the lines can be scored.
"""

import dataclasses
import datetime as dt

import pandas as pd
import pytest

from src.backtests.replay import replay_lines
from src.serving.slate import build_slate
from tests.league_fixture import make_league
from tests.slate_fixture import AS_OF, SLATE_DATE, make_log, make_models, make_roster

SAMPLE_START = dt.date(2025, 1, 1)


def _history():
    return make_league(n_games=13, teams=("BOS", "DET"))


def _log_all():
    return pd.concat([
        make_log(event="e1"),
        make_log(home="Nowhere Nomads", event="eU"),
        make_log(books=("dk", "fd"), event="eT"),
        make_log(tip="2025-02-10T23:30:00+00:00", event="eF"),
        make_log(players=("Nobody Real",), event="eN"),
    ], ignore_index=True)


def _replay(log=None, players=None, models=None, **kw):
    hp, ht = _history()
    kw.setdefault("first_game_date", SAMPLE_START)
    return replay_lines(log if log is not None else _log_all(), players if players is not None else hp, ht,
                        models or make_models(), **kw)


def _status(frame, event, player):
    return frame[(frame["event_id"] == event) & (frame["player"] == player)]["status"].iloc[0]


def test_every_kind_of_line_gets_its_own_status():
    hp, _ = _history()
    hp = hp[~((hp["GAME_ID"] == "g13") & (hp["PLAYER_ID"] == 2))]   # P2 did not play that night
    out = _replay(players=hp)
    assert _status(out, "e1", "P1") == "scored"
    assert _status(out, "e1", "P2") == "did not play"
    assert (out[out["event_id"] == "eU"]["status"] == "unknown team name").all()
    assert (out[out["event_id"] == "eT"]["status"] == "thin market").all()
    assert (out[out["event_id"] == "eF"]["status"] == "game not played yet").all()
    assert _status(out, "eN", "Nobody Real") == "name not found"


def test_the_sample_start_and_the_training_window_are_enforced():
    late = _replay(first_game_date=dt.date(2025, 1, 26))
    assert (late[late["event_id"] == "e1"]["status"] == "before sample start").all()
    trained = dataclasses.replace(make_models(), trained_through="2025-01-25")
    inside = _replay(models=trained)
    assert (inside[inside["event_id"] == "e1"]["status"] == "in training window").all()


def test_players_with_too_few_prior_games_are_not_scored():
    out = _replay(min_prior=13)
    assert (out[out["event_id"] == "e1"]["status"] == "few prior games").all()


def test_scored_lines_carry_the_actual_result_and_one_row_per_consensus_line():
    out = _replay()
    scored = out[out["status"] == "scored"].set_index("player")
    assert scored.loc["P1", "actual"] == 39 and scored.loc["P4", "actual"] == 37   # box-score points
    assert scored.loc["P1", "line"] == 10.5 and scored.loc["P1", "model_label"] == "test"
    assert len(out) == 4 + 4 + 4 + 4 + 1
    assert out.loc[out["status"] != "scored", ["projection", "actual"]].isna().all().all()


def test_the_replay_projects_exactly_what_the_live_slate_would_have():
    hp, ht = _history()
    before = hp["GAME_DATE"] < pd.Timestamp(SLATE_DATE)
    live = build_slate(make_log(), make_roster(), hp[before], ht[ht["GAME_DATE"] < pd.Timestamp(SLATE_DATE)],
                       make_models(), as_of=AS_OF, slate_date=SLATE_DATE).slate.set_index("player")
    out = _replay(log=make_log()).set_index("player")
    for player in ("P1", "P3", "P4"):
        assert out.loc[player, "projection"] == pytest.approx(live.loc[player, "projection"])
        assert out.loc[player, "min_pred"] == pytest.approx(live.loc[player, "min_pred"])


def test_a_games_own_box_score_cannot_change_its_projection():
    base = _replay(log=make_log()).set_index("player")
    hp, _ = _history()
    hp = hp.copy()
    hp.loc[(hp["GAME_ID"] == "g13") & (hp["PLAYER_ID"] == 1), "PTS"] += 50
    bumped = _replay(log=make_log(), players=hp).set_index("player")
    assert bumped.loc["P1", "actual"] == base.loc["P1", "actual"] + 50
    assert bumped.loc["P1", "projection"] == pytest.approx(base.loc["P1", "projection"])

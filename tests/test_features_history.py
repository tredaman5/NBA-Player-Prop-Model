import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.features.builder import build_features
from src.features.history import load_history
from tests.league_fixture import make_league


def _season(prefix, start):
    p, t = make_league(n_games=6)
    shift = pd.Timestamp(start) - pd.Timestamp("2025-01-01")
    for frame in (p, t):
        frame["GAME_DATE"] = frame["GAME_DATE"] + shift
        frame["GAME_ID"] = prefix + frame["GAME_ID"]
    return p, t


def _fetchers(**seasons):
    def fetch_players(season, season_type="Regular Season", use_cache=True):
        return seasons[season][0]

    def fetch_teams(season, season_type="Regular Season", use_cache=True):
        return seasons[season][1]

    return fetch_players, fetch_teams


def test_seasons_are_concatenated_in_date_order():
    s1, s2 = _season("a", "2025-01-01"), _season("b", "2025-10-21")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s2})
    players, teams = load_history(["2025-26", "2024-25"], fetch_players=fp, fetch_teams=ft)
    assert len(players) == len(s1[0]) + len(s2[0])
    assert len(teams) == len(s1[1]) + len(s2[1])
    assert players["GAME_DATE"].is_monotonic_increasing


def test_the_same_game_in_two_seasons_is_rejected():
    s1 = _season("a", "2025-01-01")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s1})
    with pytest.raises(DataQualityError, match="duplicate"):
        load_history(["2024-25", "2025-26"], fetch_players=fp, fetch_teams=ft)


def test_no_seasons_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        load_history([])


def test_features_carry_across_the_season_boundary():
    s1, s2 = _season("a", "2025-01-01"), _season("b", "2025-10-21")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s2})
    players, teams = load_history(["2024-25", "2025-26"], fetch_players=fp, fetch_teams=ft)
    f = build_features(players, teams)
    first_of_new_season = f[(f["PLAYER_ID"] == 1) & (f["GAME_ID"] == "bg1")].iloc[0]
    assert first_of_new_season["DAYS_REST"] == 14  # the offseason, capped
    prior_three = f[(f["PLAYER_ID"] == 1) & f["GAME_ID"].isin(["ag4", "ag5", "ag6"])]["MIN"]
    assert first_of_new_season["MIN_ROLL_3"] == pytest.approx(prior_three.mean())

import numpy as np
import pandas as pd

from src.features.builder import build_features
from tests.league_fixture import make_league


def _row(features, player, game):
    return features[(features["PLAYER_ID"] == player) & (features["GAME_ID"] == game)].iloc[0]


def _player_games(player):
    p, _ = make_league()
    return p[p["PLAYER_ID"] == player].sort_values("GAME_DATE").reset_index(drop=True)


def test_home_and_opponent_come_from_the_matchup_string():
    p, t = make_league()
    f = build_features(p, t)
    assert _row(f, 1, "g1")["IS_HOME"] == 1 and _row(f, 1, "g1")["OPP_ABBREVIATION"] == "B"  # A vs. B
    assert _row(f, 1, "g2")["IS_HOME"] == 0 and _row(f, 1, "g2")["OPP_ABBREVIATION"] == "B"  # A @ B


def test_days_rest_is_three_for_a_first_game_then_the_gap_in_days():
    p, t = make_league()
    f = build_features(p, t)
    assert _row(f, 1, "g1")["DAYS_REST"] == 3
    assert _row(f, 1, "g2")["DAYS_REST"] == 2
    assert _row(f, 1, "g2")["IS_B2B"] == 0


def test_rolling_minutes_use_only_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    assert np.isnan(_row(f, 1, "g1")["MIN_ROLL_3"])
    assert _row(f, 1, "g2")["MIN_ROLL_3"] == games["MIN"].iloc[0]
    # fifth game: the three games before it, not including itself
    assert _row(f, 1, "g5")["MIN_ROLL_3"] == games["MIN"].iloc[1:4].mean()
    assert _row(f, 1, "g5")["MIN_ROLL_5"] == games["MIN"].iloc[0:4].mean()  # only 4 prior games exist


def test_per_minute_rolling_rates_use_only_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    rates = (games["PTS"] / games["MIN"]).tolist()
    assert np.isclose(_row(f, 1, "g4")["PTS_PER_MIN_ROLL_3"], np.mean(rates[0:3]))
    fga_rates = (games["FGA"] / games["MIN"]).tolist()
    assert np.isclose(_row(f, 1, "g4")["FGA_PER_MIN_ROLL_3"], np.mean(fga_rates[0:3]))


def test_a_zero_minute_game_counts_as_a_zero_rate_not_a_crash():
    p, t = make_league()
    p.loc[(p["PLAYER_ID"] == 1) & (p["GAME_ID"] == "g2"), ["MIN", "PTS", "FGA"]] = 0
    f = build_features(p, t)
    first = _player_games(1)
    assert np.isclose(_row(f, 1, "g3")["PTS_PER_MIN_ROLL_3"], np.mean([first["PTS"].iloc[0] / first["MIN"].iloc[0], 0.0]))

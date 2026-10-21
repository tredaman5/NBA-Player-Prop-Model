import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features
from src.features.columns import BUILDER_FEATURES
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


def test_builder_returns_one_row_per_player_game():
    p, t = make_league()
    assert len(build_features(p, t)) == len(p)


def test_every_builder_feature_is_produced():
    p, t = make_league()
    assert set(BUILDER_FEATURES) <= set(build_features(p, t).columns)


def test_team_points_rolling_uses_the_teams_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    a = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE")["PTS"].tolist()
    assert np.isclose(_row(f, 1, "g4")["TEAM_PTS_ROLL_5"], np.mean(a[0:3]))
    assert np.isnan(_row(f, 1, "g1")["TEAM_PTS_ROLL_5"])


def test_opponent_defense_is_what_the_opponent_allowed_in_its_prior_games():
    # A player on A faces B. In every earlier game B's opponent was A, so what B allowed is what A scored.
    p, t = make_league()
    f = build_features(p, t)
    a_pts = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE")["PTS"].tolist()
    b_pts = t[t["TEAM_ABBREVIATION"] == "B"].sort_values("GAME_DATE")["PTS"].tolist()
    assert np.isclose(_row(f, 1, "g4")["OPP_PTS_ALLOWED_ROLL_5"], np.mean(a_pts[0:3]))
    assert np.isclose(_row(f, 3, "g4")["OPP_PTS_ALLOWED_ROLL_5"], np.mean(b_pts[0:3]))


def test_usage_rate_rolling_is_the_players_prior_share_of_team_possessions():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    team_a = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE").reset_index(drop=True)
    usage = [
        (g.FGA + 0.44 * g.FTA + g.TOV) / (tm.FGA + 0.44 * tm.FTA + tm.TOV)
        for g, tm in zip(games.itertuples(), team_a.itertuples())
    ]
    assert np.isclose(_row(f, 1, "g4")["USAGE_RATE_ROLL_3"], np.mean(usage[0:3]))


def test_changing_a_games_own_box_score_never_changes_its_own_features():
    """The leakage guard: tonight's result must not leak into tonight's features."""
    p, t = make_league()
    before = build_features(p, t)
    p2, t2 = p.copy(), t.copy()
    for frame, cols in (
        (p2, ["MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"]),
        (t2, ["PTS", "REB", "AST", "FGA", "FTA", "TOV"]),
    ):
        mask = frame["GAME_ID"] == "g12"
        frame.loc[mask, cols] = frame.loc[mask, cols] * 3 + 7
    after = build_features(p2, t2)
    a = before[before["GAME_ID"] == "g12"].reset_index(drop=True)[BUILDER_FEATURES]
    b = after[after["GAME_ID"] == "g12"].reset_index(drop=True)[BUILDER_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_team_rows_fail_loudly_instead_of_duplicating_players():
    p, t = make_league()
    t = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="row count"):
        build_features(p, t)

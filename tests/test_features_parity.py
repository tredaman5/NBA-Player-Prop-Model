"""
The parity test: the features computed for a game as if it were UPCOMING (history cut off
before it, the game supplied as placeholder rows) must equal the features training computes
for that same game. This is what stops serving from drifting away from training.
"""

import pandas as pd
import pytest

from src.features.builder import build_features
from src.features.columns import BUILDER_FEATURES
from tests.league_fixture import make_league


def _split_for_serving(p, t, game_id):
    """History strictly before the game, plus the game described as upcoming."""
    date = p.loc[p["GAME_ID"] == game_id, "GAME_DATE"].iloc[0]
    team_rows = t[t["GAME_ID"] == game_id]
    is_home = team_rows["MATCHUP"].str.contains("vs.", regex=False)
    home, away = team_rows[is_home].iloc[0], team_rows[~is_home].iloc[0]
    games = pd.DataFrame(
        [{
            "GAME_ID": game_id, "GAME_DATE": date,
            "HOME_TEAM_ID": home["TEAM_ID"], "HOME_ABBR": home["TEAM_ABBREVIATION"],
            "AWAY_TEAM_ID": away["TEAM_ID"], "AWAY_ABBR": away["TEAM_ABBREVIATION"],
        }]
    )
    players = p.loc[p["GAME_ID"] == game_id, ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    return p[p["GAME_DATE"] < date], t[t["GAME_DATE"] < date], games, players


def _as_upcoming(p, t, game_id):
    return build_features(*_split_for_serving(p, t, game_id))


@pytest.mark.parametrize("game_id", ["g2", "g6", "g12"])
def test_serving_features_equal_training_features(game_id):
    p, t = make_league(n_games=12)
    training = build_features(p, t)
    serving = _as_upcoming(p, t, game_id)
    a = training[training["GAME_ID"] == game_id].sort_values("PLAYER_ID").reset_index(drop=True)[BUILDER_FEATURES]
    b = serving[serving["IS_UPCOMING"]].sort_values("PLAYER_ID").reset_index(drop=True)[BUILDER_FEATURES]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_parity_holds_across_an_offseason_gap():
    p, t = make_league(n_games=12, gap_after=6, gap_days=150)
    first_after_gap = "g7"
    training = build_features(p, t)
    serving = _as_upcoming(p, t, first_after_gap)
    a = training[training["GAME_ID"] == first_after_gap].sort_values("PLAYER_ID").reset_index(drop=True)
    b = serving[serving["IS_UPCOMING"]].sort_values("PLAYER_ID").reset_index(drop=True)
    assert (a["DAYS_REST"] == 14).all() and (b["DAYS_REST"] == 14).all()  # long rest is capped, not dropped
    assert a["MIN_ROLL_3"].notna().all()  # rolling windows carry over the gap
    pd.testing.assert_frame_equal(a[BUILDER_FEATURES], b[BUILDER_FEATURES], check_dtype=False)


def test_upcoming_rows_are_flagged_and_have_unknown_results():
    p, t = make_league()
    f = _as_upcoming(p, t, "g12")
    up = f[f["IS_UPCOMING"]]
    assert len(up) == 4  # two teams, two players each
    assert up[["MIN", "PTS", "REB", "AST"]].isna().all().all()
    assert not f[~f["IS_UPCOMING"]]["MIN"].isna().any()


def test_upcoming_games_never_change_the_history_features():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    alone = build_features(hist_p, hist_t)
    combined = build_features(hist_p, hist_t, games, players)
    combined_hist = combined[~combined["IS_UPCOMING"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone[BUILDER_FEATURES], combined_hist[BUILDER_FEATURES], check_dtype=False)


def test_games_and_players_must_be_passed_together():
    p, t = make_league()
    hist_p, hist_t, games, _ = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="together"):
        build_features(hist_p, hist_t, upcoming_games=games)


def test_a_player_in_two_upcoming_games_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="more than one"):
        build_features(hist_p, hist_t, games, pd.concat([players, players.iloc[[0]]]))


def test_an_upcoming_game_already_in_history_is_rejected():
    p, t = make_league()
    _, _, games, players = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="already in the history"):
        build_features(p, t, games, players)  # full history still contains g12


def test_a_player_whose_team_is_not_in_the_game_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    players = players.copy()
    players.iloc[0, players.columns.get_loc("TEAM_ABBREVIATION")] = "ZZZ"
    with pytest.raises(ValueError, match="not playing"):
        build_features(hist_p, hist_t, games, players)


def test_an_upcoming_game_dated_before_the_end_of_history_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    games = games.copy()
    games["GAME_DATE"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="before the end of the history"):
        build_features(hist_p, hist_t, games, players)

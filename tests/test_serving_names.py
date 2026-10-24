import pandas as pd

from src.common.teams import TEAM_ABBR
from src.serving.names import match_to_roster, normalize_name


def test_all_thirty_teams_are_known_including_both_clippers_spellings():
    assert len(set(TEAM_ABBR.values())) == 30
    assert TEAM_ABBR["LA Clippers"] == TEAM_ABBR["Los Angeles Clippers"] == "LAC"


def test_normalize_name_ignores_accents_suffixes_and_punctuation():
    assert normalize_name("Nikola Jokić") == normalize_name("Nikola Jokic") == "nikola jokic"
    assert normalize_name("Jaren Jackson Jr.") == normalize_name("Jaren Jackson") == "jaren jackson"
    assert normalize_name("De'Aaron Fox") == "deaaron fox"
    assert normalize_name("Karl-Anthony Towns") == "karl anthony towns"


def _roster():
    return pd.DataFrame(
        [
            {"PERSON_ID": 1, "DISPLAY_FIRST_LAST": "Nikola Jokić", "TEAM_ID": 10, "TEAM_ABBREVIATION": "DEN"},
            {"PERSON_ID": 2, "DISPLAY_FIRST_LAST": "Jalen Williams", "TEAM_ID": 20, "TEAM_ABBREVIATION": "OKC"},
            {"PERSON_ID": 3, "DISPLAY_FIRST_LAST": "Jalen Williams", "TEAM_ID": 30, "TEAM_ABBREVIATION": "PHX"},
            {"PERSON_ID": 4, "DISPLAY_FIRST_LAST": "Traded Guy", "TEAM_ID": 30, "TEAM_ABBREVIATION": "PHX"},
        ]
    )


def _lines(rows):
    return pd.DataFrame(rows, columns=["player", "home_abbr", "away_abbr"])


def test_a_player_is_matched_to_the_team_in_the_event():
    out = match_to_roster(_lines([("Nikola Jokic", "DEN", "LAL")]), _roster())
    row = out.iloc[0]
    assert row["match_status"] == "matched" and row["PLAYER_ID"] == 1 and row["TEAM_ABBREVIATION"] == "DEN"


def test_same_name_on_different_teams_resolves_by_the_events_teams():
    out = match_to_roster(_lines([("Jalen Williams", "PHX", "LAL"), ("Jalen Williams", "OKC", "LAL")]), _roster())
    assert out["PLAYER_ID"].tolist() == [3, 2]


def test_a_player_whose_team_is_not_playing_is_not_on_either_roster():
    out = match_to_roster(_lines([("Traded Guy", "DEN", "LAL")]), _roster())
    assert out.iloc[0]["match_status"] == "not_on_either_roster"
    assert pd.isna(out.iloc[0]["PLAYER_ID"])


def test_an_unknown_name_is_not_on_either_roster():
    out = match_to_roster(_lines([("Totally Made Up", "DEN", "LAL")]), _roster())
    assert out.iloc[0]["match_status"] == "not_on_either_roster"


def test_two_candidates_on_the_events_teams_is_ambiguous():
    out = match_to_roster(_lines([("Jalen Williams", "OKC", "PHX")]), _roster())
    assert out.iloc[0]["match_status"] == "ambiguous"


def test_every_input_line_comes_back_exactly_once():
    lines = _lines([("Nikola Jokic", "DEN", "LAL"), ("Jalen Williams", "OKC", "PHX"), ("Nobody", "DEN", "LAL")])
    assert len(match_to_roster(lines, _roster())) == len(lines)

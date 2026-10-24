import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.serving.rosters import validate_roster


def _roster(n_teams=30, per_team=3):
    rows = []
    for t in range(n_teams):
        for k in range(per_team):
            rows.append({"PERSON_ID": t * 100 + k, "DISPLAY_FIRST_LAST": f"P{t}-{k}",
                         "TEAM_ID": 1000 + t, "TEAM_ABBREVIATION": f"T{t:02d}"})
    return pd.DataFrame(rows)


def test_a_healthy_roster_passes():
    validate_roster(_roster())


def test_too_few_teams_means_the_pull_was_partial():
    with pytest.raises(DataQualityError, match="teams"):
        validate_roster(_roster(n_teams=12))


def test_a_player_listed_twice_is_rejected():
    r = _roster()
    with pytest.raises(DataQualityError, match="duplicate"):
        validate_roster(pd.concat([r, r.iloc[[0]]], ignore_index=True))


def test_missing_columns_are_rejected():
    with pytest.raises(DataQualityError, match="columns"):
        validate_roster(_roster().drop(columns=["TEAM_ABBREVIATION"]))


def test_an_empty_roster_is_rejected():
    with pytest.raises(DataQualityError, match="empty"):
        validate_roster(_roster().iloc[0:0])

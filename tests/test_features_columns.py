from src.features.columns import BUILDER_FEATURES, MINUTES_FEATURES, PROP_FEATURES, PROP_TARGETS

BANNED = {"MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"}  # tonight's own box score


def test_no_model_uses_the_current_games_box_score():
    for name, feats in {"minutes": MINUTES_FEATURES, **PROP_FEATURES}.items():
        assert not BANNED & set(feats), f"{name} uses a current-game stat"


def test_every_model_feature_comes_from_the_builder_or_is_min_pred():
    for name, feats in {"minutes": MINUTES_FEATURES, **PROP_FEATURES}.items():
        unknown = set(feats) - set(BUILDER_FEATURES) - {"MIN_PRED"}
        assert not unknown, f"{name} asks for columns the builder never makes: {unknown}"


def test_current_model_shapes_are_preserved():
    assert len(PROP_FEATURES["points"]) == 26
    assert len(PROP_FEATURES["rebounds"]) == 15
    assert len(PROP_FEATURES["assists"]) == 15
    assert "TEAM_PTS_ROLL_5" in PROP_FEATURES["points"]
    assert "TEAM_PTS_ROLL_5" not in PROP_FEATURES["rebounds"]  # only the points model ever had it
    assert PROP_TARGETS == {"points": "PTS", "rebounds": "REB", "assists": "AST"}

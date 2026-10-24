import json

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.serving.models import MARKET_TO_PROP, load_models


def _write_set(folder, label, rmse=2.0, trained_through="2026-03-11", skip=()):
    for name in ("minutes", "points", "rebounds", "assists"):
        if name in skip:
            continue
        model = DummyRegressor(strategy="constant", constant=1.0).fit(np.zeros((2, 1)), [1.0, 1.0])
        joblib.dump(model, folder / f"{name}_model_{label}.joblib")
        (folder / f"{name}_features_{label}.json").write_text(json.dumps([f"{name}_feat"]))
        (folder / f"{name}_eval_{label}.json").write_text(json.dumps({"rmse": rmse, "trained_through": trained_through}))


def test_market_names_map_to_props():
    assert MARKET_TO_PROP == {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists"}


def test_the_newest_complete_set_is_loaded(tmp_path):
    _write_set(tmp_path, "2024-25_RegularSeason", rmse=1.0)
    _write_set(tmp_path, "2025-26_RegularSeason", rmse=2.5, trained_through="2026-03-11")
    m = load_models(tmp_path)
    assert m.label == "2025-26_RegularSeason"
    assert m.trained_through == "2026-03-11"
    assert m.rmse["points"] == 2.5
    assert m.features["points"] == ["points_feat"] and m.features["minutes"] == ["minutes_feat"]
    assert set(m.props) == {"points", "rebounds", "assists"}


def test_an_incomplete_newer_set_falls_back_to_the_newest_complete_one(tmp_path):
    _write_set(tmp_path, "2024-25_RegularSeason")
    _write_set(tmp_path, "2025-26_RegularSeason", skip=("assists",))
    assert load_models(tmp_path).label == "2024-25_RegularSeason"


def test_an_explicit_label_with_a_missing_file_names_what_is_missing(tmp_path):
    _write_set(tmp_path, "2025-26_RegularSeason", skip=("assists",))
    with pytest.raises(FileNotFoundError, match="assists_model_2025-26_RegularSeason"):
        load_models(tmp_path, label="2025-26_RegularSeason")


def test_no_artifacts_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no complete set"):
        load_models(tmp_path)

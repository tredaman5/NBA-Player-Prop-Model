import dataclasses
import datetime as dt
import json

import pytest

from src.common.validation import DataQualityError
from src.pipeline import daily
from src.pipeline.checks import NoLinesError
from tests.slate_fixture import AS_OF, SLATE_DATE, make_history, make_log, make_models, make_roster


def _run(tmp_path, log=None, models=None, history_calls=None, **kw):
    log_path = tmp_path / "lines.csv"
    (log if log is not None else make_log()).to_csv(log_path, index=False)

    def load_history_fn(seasons, refresh_seasons=()):
        if history_calls is not None:
            history_calls.append((list(seasons), list(refresh_seasons)))
        return make_history()

    kw.setdefault("slate_date", SLATE_DATE)
    return daily.run_pipeline(
        log_path=log_path, out_dir=tmp_path / "out", as_of=AS_OF,
        load_history_fn=load_history_fn, fetch_roster_fn=lambda season: make_roster(),
        load_models_fn=lambda: models or make_models(), **kw,
    )


def test_a_good_run_writes_the_slate_and_a_summary(tmp_path):
    summary, result = _run(tmp_path)
    assert summary["status"] == "ok"
    assert (summary["n_lines"], summary["n_projected"], summary["n_skipped"]) == (4, 4, 0)
    assert (tmp_path / "out" / "slate_2025-01-25.csv").exists()
    assert "P1" in (tmp_path / "out" / "slate_2025-01-25.html").read_text()
    on_disk = json.loads((tmp_path / "out" / "run_2025-01-25.json").read_text())
    assert on_disk["status"] == "ok" and on_disk["history_last_date"] == "2025-01-23"
    assert len(result.slate) == 4


def test_history_covers_this_and_last_season_and_refreshes_only_this_one(tmp_path):
    calls = []
    _run(tmp_path, history_calls=calls)
    assert calls == [(["2023-24", "2024-25"], ["2024-25"])]


def test_skipped_reasons_are_counted(tmp_path):
    # P5 is on neither roster: 1 of 5 lines, exactly the 20% limit, so the run still passes
    summary, _ = _run(tmp_path, log=make_log(players=("P1", "P2", "P3", "P4", "P5")))
    assert summary["skipped_by_reason"] == {"not on either team's roster": 1}


def test_a_failed_check_writes_a_failed_summary_and_raises(tmp_path):
    with pytest.raises(DataQualityError, match="matched"):
        _run(tmp_path, log=make_log(players=("Nobody A", "Nobody B")))
    on_disk = json.loads((tmp_path / "out" / "run_2025-01-25.json").read_text())
    assert on_disk["status"] == "failed" and "matched" in on_disk["error"]
    assert not (tmp_path / "out" / "slate_2025-01-25.csv").exists()


def test_a_date_with_no_games_raises_no_lines(tmp_path):
    with pytest.raises(NoLinesError):
        _run(tmp_path, slate_date=dt.date(2025, 1, 26))


def test_old_models_are_a_warning_not_a_failure(tmp_path):
    old = dataclasses.replace(make_models(), trained_through="2024-01-01")
    summary, _ = _run(tmp_path, models=old)
    assert summary["status"] == "ok" and any("retrain" in w for w in summary["warnings"])


def test_exit_codes(monkeypatch, tmp_path, capsys):
    def boom(exc):
        def _raise(*a, **k):
            raise exc
        return _raise

    argv = ["--date", "2025-01-25", "--log-path", str(tmp_path / "x.csv")]
    monkeypatch.setattr(daily, "run_pipeline", boom(NoLinesError("no games")))
    assert daily.main(argv) == 2
    monkeypatch.setattr(daily, "run_pipeline", boom(DataQualityError("broken")))
    assert daily.main(argv) == 1
    assert "broken" in capsys.readouterr().err


def test_models_dir_is_passed_to_the_model_loader(monkeypatch, tmp_path):
    seen = {}

    def fake_load_models(artifacts_dir=None, label=None):
        seen["dir"] = artifacts_dir
        return make_models()

    monkeypatch.setattr("src.serving.models.load_models", fake_load_models)
    log_path = tmp_path / "lines.csv"
    make_log().to_csv(log_path, index=False)
    daily.run_pipeline(
        SLATE_DATE, log_path, out_dir=tmp_path / "out", as_of=AS_OF, models_dir=tmp_path / "m",
        load_history_fn=lambda seasons, refresh_seasons=(): make_history(),
        fetch_roster_fn=lambda season: make_roster(),
    )
    assert seen["dir"] == tmp_path / "m"

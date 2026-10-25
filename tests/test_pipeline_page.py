import re

from src.pipeline.page import render_page
from src.serving.slate import build_slate
from tests.slate_fixture import AS_OF, SLATE_DATE, make_history, make_log, make_models, make_roster

SUMMARY = {"warnings": ["models were trained through 2025-01-01, 100 days before this slate; retrain"],
           "as_of": "2025-01-25 12:00:00+00:00"}


def _result(**kw):
    hp, ht = make_history()
    return build_slate(kw.pop("log", make_log()), make_roster(), hp, ht, make_models(),
                       as_of=AS_OF, slate_date=SLATE_DATE, **kw)


def test_the_page_lists_players_with_line_projection_and_difference():
    html = render_page(_result(), SUMMARY)
    assert html.lstrip().lower().startswith("<!doctype html>")
    for text in ("P1", "P4", "10.5", "6:30 PM", "BOS", "DET", "<table"):
        assert text in html


def test_names_are_escaped_so_a_hostile_name_cannot_inject_markup():
    log = make_log(players=("P1", "P2", "P3", "P4", "<script>alert(1)</script>"))
    html = render_page(_result(log=log), SUMMARY)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_warnings_and_skip_reasons_are_shown():
    log = make_log(players=("P1", "P2", "P3", "P4", "Nobody Real"))
    html = render_page(_result(log=log), SUMMARY)
    assert "retrain" in html
    assert "Nobody Real" in html and "not on either team" in html


def test_no_betting_language_and_no_external_requests():
    html = render_page(_result(), SUMMARY)
    assert not re.search(r"\b(edge|bet|ev|kelly|probability|pick)s?\b", html, re.I)
    assert not re.search(r"(src|href)=[\"']https?://", html)


def test_an_empty_slate_still_renders():
    html = render_page(_result(log=make_log(players=("Nobody Real",))), SUMMARY)
    assert "No projections" in html


def test_a_tiny_negative_difference_does_not_show_as_negative_zero():
    result = _result()
    result.slate.loc[0, "difference"] = -0.04
    html = render_page(result, SUMMARY)
    assert "-0.0<" not in html and "+0.0<" in html


def _backtest(**kw):
    base = {"model_label": "test", "verdict": "interim", "n_directional": 212, "lines_needed": 1288,
            "hit_rate": 0.55, "hit_ci": [0.48, 0.62], "mean_d": 0.1, "d_ci": [-0.1, 0.3], "n_scored": 220}
    return {**base, **kw}


def test_an_interim_backtest_line_says_it_is_not_a_verdict_and_uses_no_betting_language():
    html = render_page(_result(), {**SUMMARY, "backtest": _backtest()})
    assert "Backtest (interim, not a verdict)" in html and "212 of 1,500" in html and "55%" in html
    assert not re.search(r"\b(edge|bet|ev|kelly|probability|pick)s?\b", html, re.I)


def test_a_no_edge_result_is_stated_without_hype():
    html = render_page(_result(), {**SUMMARY, "backtest": _backtest(verdict="no_edge_shown", n_directional=1500)})
    assert "did not beat the lines" in html


def test_a_page_without_backtest_data_still_renders():
    assert "Backtest" not in render_page(_result(), SUMMARY)

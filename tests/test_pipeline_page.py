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

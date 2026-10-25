import re

from src.backtests.metrics import summarize
from src.backtests.report import render_report
from tests.test_backtest_metrics import _frame, _synthetic


def test_an_interim_report_says_so_and_uses_no_betting_language():
    text = render_report(summarize(_frame([("e1", 10.5, 12.0, 15), ("e2", 10.5, 9.0, 12)])))
    assert "INTERIM, not a verdict" in text and "of 1,500 lines needed" in text
    assert not re.search(r"\b(edge|bet|ev|pick)s?\b", text, re.I)


def test_a_report_with_nothing_scored_explains_why():
    text = render_report(summarize(_frame([("e1", 10.5, 12.0, 15)]).iloc[0:0]))
    assert "No scored lines yet" in text


def test_the_verdicts_are_stated_plainly():
    assert "RESULT: edge shown" in render_report(summarize(_synthetic(1500)))
    assert "RESULT: no edge shown" in render_report(summarize(_synthetic(1500, hit_every=2)))


def test_a_dry_run_is_labelled_as_not_a_result():
    assert render_report(summarize(_synthetic(10)), dry_run=True).startswith("DRY RUN")

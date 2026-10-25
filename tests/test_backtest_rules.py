"""The constants in code must say what the pre-registered rules file says."""

import datetime as dt
import re
from pathlib import Path

from src.backtests import rules

TEXT = (Path(__file__).resolve().parents[1] / "design" / "backtest-rules.md").read_text()


def test_the_rules_file_states_every_number_the_code_uses():
    assert "2026-10-26" in TEXT and rules.FIRST_GAME_DATE == dt.date(2026, 10, 26)
    assert "1,500" in TEXT and rules.VERDICT_MIN_LINES == 1500
    assert "0.524" in TEXT and rules.BREAKEVEN == 0.524
    assert "5,000 draws" in TEXT and rules.N_BOOT == 5000
    assert "seed 2026" in TEXT and rules.BOOT_SEED == 2026
    assert "2.5th and 97.5th" in TEXT and rules.CI_PERCENTILES == (2.5, 97.5)
    assert re.search(r"at least 5 prior games", TEXT) and rules.MIN_PRIOR_GAMES == 5


def test_gap_buckets_match_the_file():
    assert "under 1, 1 to 2, 2 to 3, 3 or more" in TEXT
    assert rules.GAP_BUCKETS == ((0, 1), (1, 2), (2, 3), (3, float("inf")))

"""
rules.py: the backtest's pre-registered numbers.

Copied from design/backtest-rules.md (version 1, 2026-10-25); tests/test_backtest_rules.py checks that
the two agree. Change a number only by adding a new dated version of that file first, before looking
at any result under it.
"""

import datetime as dt

FIRST_GAME_DATE = dt.date(2026, 10, 26)   # games before this were looked at informally and are excluded
MIN_PRIOR_GAMES = 5
VERDICT_MIN_LINES = 1500                  # non-push lines with a direction, for one model label
BREAKEVEN = 0.524                         # break-even hit rate at -110 on both sides
N_BOOT = 5000
BOOT_SEED = 2026
CI_PERCENTILES = (2.5, 97.5)
GAP_BUCKETS = ((0, 1), (1, 2), (2, 3), (3, float("inf")))   # exploratory only

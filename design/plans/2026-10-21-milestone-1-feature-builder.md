# Milestone 1: Unified Feature Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One pure feature builder that both training and (later) serving use, proven equal by a parity test, over a continuous multi-season history, and a trainer that uses it to retrain on both seasons.

**Architecture:** `src/features/builder.py` turns a player-game log, a team-game log, and optionally placeholder rows for upcoming games into one feature table. Upcoming games enter as rows with unknown box scores, and the existing `shift(1)` rolling logic means they see only the past. `src/training/train_all.py` builds features over several seasons of history, holds out the most recent season's last 20% by date, and writes the same artifact files the current pipeline reads.

**Tech Stack:** pandas, numpy, scikit-learn (`HistGradientBoostingRegressor`), joblib, pytest. No new dependencies.

**Spec:** `design/2026-10-21-slate-rewrite.md` (milestone 1 of 6).

Commands assume the project virtualenv is active (`pip install -r requirements.txt`) and are run from the repo root.

## Global Constraints

- Training and serving share one code path: no feature logic outside `src/features/builder.py`.
- Every rolling feature uses only games strictly before the game it describes (`shift(1)`).
- Holdouts are chronological and by date, never random.
- The feature builder does no file or network I/O (pure functions over DataFrames).
- Every data pull goes through `src/common/validation.py` before it is used.
- Per-model feature lists stay exactly as the current models use them (rebounds and assists have no `TEAM_PTS_ROLL_*`).
- Old per-season modules (`src/minutes/*`, `src/props/*`) stay untouched in this milestone; they retire in milestone 4.
- No new third-party dependencies.
- Specs and plans live in `design/`, never `docs/` (`docs/` is the public Pages root).
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/common/seasons.py` (create) | `season_label`, `season_for_date` |
| `src/features/__init__.py` (create) | package marker |
| `src/features/columns.py` (create) | feature-name lists per model, built from the same window constants |
| `src/features/builder.py` (create) | `build_features(...)`: the one feature builder |
| `src/features/history.py` (create) | `load_history(seasons)`: continuous multi-season logs, validated |
| `src/training/__init__.py` (create) | package marker |
| `src/training/train_all.py` (create) | `holdout_cutoff`, `train_models`, `train_all` |
| `main.py` (modify) | add the `train-all` command |
| `tests/league_fixture.py` (create) | tiny deterministic league used by feature tests |
| `tests/test_seasons.py`, `tests/test_features_columns.py`, `tests/test_features_builder.py`, `tests/test_features_parity.py`, `tests/test_features_history.py`, `tests/test_train_all.py` (create) | tests |

---

### Task 1: Season helpers and feature column lists

**Files:**
- Create: `src/common/seasons.py`, `src/features/__init__.py`, `src/features/columns.py`
- Test: `tests/test_seasons.py`, `tests/test_features_columns.py`

**Interfaces:**
- Produces: `season_label(start_year: int) -> str`, `season_for_date(date) -> str`; in `columns.py`: `MIN_WINDOWS`, `RATE_STATS`, `TEAM_PTS_WINDOWS`, `OPP_STATS`, `OPP_WINDOWS`, `USAGE_WINDOWS`, `BUILDER_FEATURES: list[str]`, `MINUTES_FEATURES`, `PROP_FEATURES: dict[str, list[str]]`, `PROP_TARGETS: dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_seasons.py`:
```python
from src.common.seasons import season_for_date, season_label


def test_season_label():
    assert season_label(2025) == "2025-26"
    assert season_label(1999) == "1999-00"


def test_season_for_date_covers_october_through_the_finals():
    assert season_for_date("2025-10-21") == "2025-26"
    assert season_for_date("2026-04-12") == "2025-26"
    assert season_for_date("2026-06-15") == "2025-26"
    assert season_for_date("2026-10-20") == "2026-27"
```

`tests/test_features_columns.py`:
```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_seasons.py tests/test_features_columns.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.common.seasons'`.

- [ ] **Step 3: Write the implementation**

`src/common/seasons.py`:
```python
"""NBA season naming helpers."""

from __future__ import annotations

import pandas as pd


def season_label(start_year: int) -> str:
    """2025 -> "2025-26"."""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def season_for_date(date) -> str:
    """The season a game date belongs to. Games run October to June, so September on starts a new one."""
    d = pd.Timestamp(date)
    return season_label(d.year if d.month >= 9 else d.year - 1)
```

`src/features/__init__.py`: empty file.

`src/features/columns.py`:
```python
"""Feature names, per model. One place, so the builder and the trainers cannot drift apart."""

from __future__ import annotations

MIN_WINDOWS = (3, 5, 10)
RATE_STATS = ("FGA", "FTA", "FG3A", "PTS", "REB", "AST")
TEAM_PTS_WINDOWS = (5, 10)
OPP_STATS = ("PTS", "REB", "AST")
OPP_WINDOWS = (5, 10)
USAGE_WINDOWS = (3, 5, 10)

_CONTEXT = ["IS_HOME", "DAYS_REST", "IS_B2B"]
_MIN_ROLL = [f"MIN_ROLL_{w}" for w in MIN_WINDOWS]
_USAGE = [f"USAGE_RATE_ROLL_{w}" for w in USAGE_WINDOWS]


def _rates(stat: str) -> list[str]:
    return [f"{stat}_PER_MIN_ROLL_{w}" for w in MIN_WINDOWS]


def _opp(stat: str) -> list[str]:
    return [f"OPP_{stat}_ALLOWED_ROLL_{w}" for w in OPP_WINDOWS]


_TEAM_PTS = [f"TEAM_PTS_ROLL_{w}" for w in TEAM_PTS_WINDOWS]

# Everything the builder emits that a model may use.
BUILDER_FEATURES = (
    _CONTEXT
    + _MIN_ROLL
    + [c for s in RATE_STATS for c in _rates(s)]
    + _USAGE
    + _TEAM_PTS
    + [c for s in OPP_STATS for c in _opp(s)]
)

MINUTES_FEATURES = _CONTEXT + _MIN_ROLL

PROP_FEATURES = {
    "points": (
        _CONTEXT + _MIN_ROLL + ["MIN_PRED"]
        + _rates("FGA") + _rates("FTA") + _rates("FG3A") + _rates("PTS")
        + _TEAM_PTS + _opp("PTS") + _USAGE
    ),
    "rebounds": _CONTEXT + _MIN_ROLL + ["MIN_PRED"] + _rates("REB") + _opp("REB") + _USAGE,
    "assists": _CONTEXT + _MIN_ROLL + ["MIN_PRED"] + _rates("AST") + _opp("AST") + _USAGE,
}

PROP_TARGETS = {"points": "PTS", "rebounds": "REB", "assists": "AST"}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_seasons.py tests/test_features_columns.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/common/seasons.py src/features/__init__.py src/features/columns.py tests/test_seasons.py tests/test_features_columns.py
git commit -m "feat: add season helpers and per-model feature column lists"
```

---

### Task 2: Builder core: context, rest, rolling minutes, per-minute rates

**Files:**
- Create: `src/features/builder.py`, `tests/league_fixture.py`
- Test: `tests/test_features_builder.py`

**Interfaces:**
- Consumes: `MIN_WINDOWS`, `RATE_STATS` from `src/features/columns.py`.
- Produces: `PLAYER_COLS`, `TEAM_COLS` (lists of required column names); `build_features(player_games: DataFrame, team_games: DataFrame) -> DataFrame` (one row per input player-game, sorted by `GAME_DATE, GAME_ID, PLAYER_ID`, with `IS_HOME, OPP_ABBREVIATION, DAYS_REST, IS_B2B, MIN_ROLL_{3,5,10}, {FGA,FTA,FG3A,PTS,REB,AST}_PER_MIN_ROLL_{3,5,10}`). Later tasks extend this function.

- [ ] **Step 1: Write the fixture and the failing tests**

`tests/league_fixture.py`:
```python
"""A tiny deterministic league for feature tests: teams A and B, two players each,
one game every two days, A and B alternating home and away."""

from __future__ import annotations

import pandas as pd

TEAM_IDS = {"A": 100, "B": 200}


def make_league(n_games: int = 12, gap_after: int | None = None, gap_days: int = 150):
    """Return (player_games, team_games). If gap_after is set, every game from that index on is
    pushed gap_days later, to mimic an offseason."""
    start = pd.Timestamp("2025-01-01")
    player_rows, team_rows = [], []
    for g in range(n_games):
        date = start + pd.Timedelta(days=2 * g)
        if gap_after is not None and g >= gap_after:
            date += pd.Timedelta(days=gap_days)
        home, away = ("A", "B") if g % 2 == 0 else ("B", "A")
        game_id = f"g{g + 1}"
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            matchup = f"{team} vs. {opp}" if is_home else f"{team} @ {opp}"
            totals = dict.fromkeys(["PTS", "REB", "AST", "FGA", "FTA", "TOV"], 0)
            for slot in (0, 1):
                pid = (1 if team == "A" else 3) + slot
                pts = 10 + 2 * g + 3 * slot + (5 if team == "A" else 0) + g % 3
                stats = {
                    "MIN": 32 - 10 * slot + g % 4,
                    "PTS": pts,
                    "REB": 3 + slot + g % 2,
                    "AST": 2 + slot + g % 3,
                    "FGA": pts - 2,
                    "FTA": 2 + g % 3,
                    "FG3A": slot + g % 2,
                    "TOV": 1 + g % 2,
                }
                player_rows.append(
                    {
                        "GAME_ID": game_id, "GAME_DATE": date, "PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}",
                        "TEAM_ID": TEAM_IDS[team], "TEAM_ABBREVIATION": team, "MATCHUP": matchup, **stats,
                    }
                )
                for key in totals:
                    totals[key] += stats[key]
            team_rows.append(
                {
                    "GAME_ID": game_id, "GAME_DATE": date, "TEAM_ID": TEAM_IDS[team],
                    "TEAM_ABBREVIATION": team, "MATCHUP": matchup, **totals,
                }
            )
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)
```

`tests/test_features_builder.py`:
```python
import numpy as np
import pandas as pd

from src.features.builder import build_features
from tests.league_fixture import make_league


def _row(features, player, game):
    return features[(features["PLAYER_ID"] == player) & (features["GAME_ID"] == game)].iloc[0]


def _player_games(player):
    p, _ = make_league()
    return p[p["PLAYER_ID"] == player].sort_values("GAME_DATE").reset_index(drop=True)


def test_home_and_opponent_come_from_the_matchup_string():
    p, t = make_league()
    f = build_features(p, t)
    assert _row(f, 1, "g1")["IS_HOME"] == 1 and _row(f, 1, "g1")["OPP_ABBREVIATION"] == "B"  # A vs. B
    assert _row(f, 1, "g2")["IS_HOME"] == 0 and _row(f, 1, "g2")["OPP_ABBREVIATION"] == "B"  # A @ B


def test_days_rest_is_three_for_a_first_game_then_the_gap_in_days():
    p, t = make_league()
    f = build_features(p, t)
    assert _row(f, 1, "g1")["DAYS_REST"] == 3
    assert _row(f, 1, "g2")["DAYS_REST"] == 2
    assert _row(f, 1, "g2")["IS_B2B"] == 0


def test_rolling_minutes_use_only_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    assert np.isnan(_row(f, 1, "g1")["MIN_ROLL_3"])
    assert _row(f, 1, "g2")["MIN_ROLL_3"] == games["MIN"].iloc[0]
    # fifth game: the three games before it, not including itself
    assert _row(f, 1, "g5")["MIN_ROLL_3"] == games["MIN"].iloc[1:4].mean()
    assert _row(f, 1, "g5")["MIN_ROLL_5"] == games["MIN"].iloc[0:4].mean()  # only 4 prior games exist


def test_per_minute_rolling_rates_use_only_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    rates = (games["PTS"] / games["MIN"]).tolist()
    assert np.isclose(_row(f, 1, "g4")["PTS_PER_MIN_ROLL_3"], np.mean(rates[0:3]))
    fga_rates = (games["FGA"] / games["MIN"]).tolist()
    assert np.isclose(_row(f, 1, "g4")["FGA_PER_MIN_ROLL_3"], np.mean(fga_rates[0:3]))


def test_a_zero_minute_game_counts_as_a_zero_rate_not_a_crash():
    p, t = make_league()
    p.loc[(p["PLAYER_ID"] == 1) & (p["GAME_ID"] == "g2"), ["MIN", "PTS", "FGA"]] = 0
    f = build_features(p, t)
    assert np.isclose(_row(f, 1, "g3")["PTS_PER_MIN_ROLL_3"], np.mean([_player_games(1)["PTS"].iloc[0] / _player_games(1)["MIN"].iloc[0], 0.0]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.builder'`.

- [ ] **Step 3: Write the implementation**

`src/features/builder.py`:
```python
"""
builder.py: the one feature builder.

Training, serving and the backtest all call build_features(), so a feature can never exist
in one and be stale in another. Pure functions over DataFrames: no file or network I/O.

Every rolling feature is a mean over the games BEFORE the one it describes (shift(1)), so
a game's features never include its own box score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.columns import MIN_WINDOWS, RATE_STATS

PLAYER_COLS = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "MATCHUP", "MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV",
]
TEAM_COLS = [
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
    "PTS", "REB", "AST", "FGA", "FTA", "TOV",
]


def _rolling_prior_mean(df: pd.DataFrame, by: str, column: str, window: int) -> pd.Series:
    """Mean of `column` over the previous `window` rows of each `by` group (current row excluded).
    df must already be sorted by (by, date)."""
    return df.groupby(by)[column].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def _add_context(p: pd.DataFrame) -> pd.DataFrame:
    """IS_HOME and OPP_ABBREVIATION from MATCHUP ("LAL vs. GSW" is home, "LAL @ GSW" is away)."""
    matchup = p["MATCHUP"].astype(str)
    p["IS_HOME"] = matchup.str.contains("vs.", regex=False).astype(int)
    p["OPP_ABBREVIATION"] = matchup.str.split().str[-1]
    return p


def _add_rest(p: pd.DataFrame) -> pd.DataFrame:
    """Days since the player's previous game (3 when there is none), capped at 14."""
    prev = p.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    p["DAYS_REST"] = (p["GAME_DATE"] - prev).dt.days.fillna(3).clip(lower=0, upper=14)
    p["IS_B2B"] = (p["DAYS_REST"] <= 1).astype(int)
    return p


def _add_rolling_minutes(p: pd.DataFrame) -> pd.DataFrame:
    p["MIN"] = pd.to_numeric(p["MIN"], errors="coerce")
    for w in MIN_WINDOWS:
        p[f"MIN_ROLL_{w}"] = _rolling_prior_mean(p, "PLAYER_ID", "MIN", w)
    return p


def _add_per_minute_rates(p: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-minute production. A 0-minute game counts as a 0 rate."""
    minutes = pd.to_numeric(p["MIN"], errors="coerce").replace(0, np.nan)
    for stat in RATE_STATS:
        rate = (pd.to_numeric(p[stat], errors="coerce") / minutes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p[f"_{stat}_RATE"] = rate
        for w in MIN_WINDOWS:
            p[f"{stat}_PER_MIN_ROLL_{w}"] = _rolling_prior_mean(p, "PLAYER_ID", f"_{stat}_RATE", w)
    return p.drop(columns=[f"_{s}_RATE" for s in RATE_STATS])


def build_features(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per input player-game with its features, sorted by (GAME_DATE, GAME_ID, PLAYER_ID)."""
    p = player_games[PLAYER_COLS].copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])

    p = _add_context(p)
    p = p.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    p = _add_rest(p)
    p = _add_rolling_minutes(p)
    p = _add_per_minute_rates(p)

    return p.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features_builder.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features/builder.py tests/league_fixture.py tests/test_features_builder.py
git commit -m "feat: add the unified feature builder core (context, rest, minutes, rates)"
```

---

### Task 3: Builder completion: team points, usage, opponent defense, row-count guard

**Files:**
- Modify: `src/features/builder.py` (imports, three new functions, replace `build_features`)
- Test: `tests/test_features_builder.py` (append)

**Interfaces:**
- Consumes: `add_usage_rate_rolling(player_df, team_gamelog, windows)` from `src/common/usage_stats.py`; `build_opponent_defense_table(team_gamelog, stat_cols, windows)`, `merge_opponent_features(player_df, defense_table, feature_cols)`, `opponent_feature_cols(stat, windows)` from `src/common/opponent_stats.py`; `OPP_STATS, OPP_WINDOWS, TEAM_PTS_WINDOWS, USAGE_WINDOWS` from `columns.py`.
- Produces: `build_features` now returns every column in `BUILDER_FEATURES`.

- [ ] **Step 1: Append the failing tests to `tests/test_features_builder.py`**

Add `import pytest` and `from src.features.columns import BUILDER_FEATURES` to the imports at the top, then append:
```python
def test_builder_returns_one_row_per_player_game():
    p, t = make_league()
    assert len(build_features(p, t)) == len(p)


def test_every_builder_feature_is_produced():
    p, t = make_league()
    assert set(BUILDER_FEATURES) <= set(build_features(p, t).columns)


def test_team_points_rolling_uses_the_teams_prior_games():
    p, t = make_league()
    f = build_features(p, t)
    a = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE")["PTS"].tolist()
    assert np.isclose(_row(f, 1, "g4")["TEAM_PTS_ROLL_5"], np.mean(a[0:3]))
    assert np.isnan(_row(f, 1, "g1")["TEAM_PTS_ROLL_5"])


def test_opponent_defense_is_what_the_opponent_allowed_in_its_prior_games():
    # A player on A faces B. In every earlier game B's opponent was A, so what B allowed is what A scored.
    p, t = make_league()
    f = build_features(p, t)
    a_pts = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE")["PTS"].tolist()
    b_pts = t[t["TEAM_ABBREVIATION"] == "B"].sort_values("GAME_DATE")["PTS"].tolist()
    assert np.isclose(_row(f, 1, "g4")["OPP_PTS_ALLOWED_ROLL_5"], np.mean(a_pts[0:3]))
    assert np.isclose(_row(f, 3, "g4")["OPP_PTS_ALLOWED_ROLL_5"], np.mean(b_pts[0:3]))


def test_usage_rate_rolling_is_the_players_prior_share_of_team_possessions():
    p, t = make_league()
    f = build_features(p, t)
    games = _player_games(1)
    team_a = t[t["TEAM_ABBREVIATION"] == "A"].sort_values("GAME_DATE").reset_index(drop=True)
    usage = [
        (g.FGA + 0.44 * g.FTA + g.TOV) / (tm.FGA + 0.44 * tm.FTA + tm.TOV)
        for g, tm in zip(games.itertuples(), team_a.itertuples())
    ]
    assert np.isclose(_row(f, 1, "g4")["USAGE_RATE_ROLL_3"], np.mean(usage[0:3]))


def test_changing_a_games_own_box_score_never_changes_its_own_features():
    """The leakage guard: tonight's result must not leak into tonight's features."""
    p, t = make_league()
    before = build_features(p, t)
    p2, t2 = p.copy(), t.copy()
    for frame, cols in (
        (p2, ["MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"]),
        (t2, ["PTS", "REB", "AST", "FGA", "FTA", "TOV"]),
    ):
        mask = frame["GAME_ID"] == "g12"
        frame.loc[mask, cols] = frame.loc[mask, cols] * 3 + 7
    after = build_features(p2, t2)
    a = before[before["GAME_ID"] == "g12"].reset_index(drop=True)[BUILDER_FEATURES]
    b = after[after["GAME_ID"] == "g12"].reset_index(drop=True)[BUILDER_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_team_rows_fail_loudly_instead_of_duplicating_players():
    p, t = make_league()
    t = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="row count"):
        build_features(p, t)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features_builder.py -v`
Expected: the new tests FAIL (missing columns such as `TEAM_PTS_ROLL_5`); the five Task 2 tests still pass.

- [ ] **Step 3: Implement. In `src/features/builder.py`:**

Replace the import line `from src.features.columns import MIN_WINDOWS, RATE_STATS` with:
```python
from src.common.opponent_stats import (
    build_opponent_defense_table,
    merge_opponent_features,
    opponent_feature_cols,
)
from src.common.usage_stats import add_usage_rate_rolling
from src.features.columns import (
    MIN_WINDOWS,
    OPP_STATS,
    OPP_WINDOWS,
    RATE_STATS,
    TEAM_PTS_WINDOWS,
    USAGE_WINDOWS,
)
```

Add these functions above `build_features`:
```python
def _add_usage(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling share of the team's possessions the player used (FGA + 0.44*FTA + TOV)."""
    return add_usage_rate_rolling(p, t, windows=USAGE_WINDOWS).reset_index(drop=True)


def _add_team_points(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of the player's own team's points in its previous games."""
    team = t[["GAME_ID", "TEAM_ID", "GAME_DATE", "PTS"]].copy()
    team["PTS"] = pd.to_numeric(team["PTS"], errors="coerce")
    team = team.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    cols = []
    for w in TEAM_PTS_WINDOWS:
        col = f"TEAM_PTS_ROLL_{w}"
        team[col] = _rolling_prior_mean(team, "TEAM_ID", "PTS", w)
        cols.append(col)
    return p.merge(team[["TEAM_ID", "GAME_ID"] + cols], on=["TEAM_ID", "GAME_ID"], how="left")


def _add_opponent(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Rolling points/rebounds/assists the player's opponent has allowed in its previous games."""
    defense = build_opponent_defense_table(t, stat_cols=OPP_STATS, windows=OPP_WINDOWS)
    cols = [c for stat in OPP_STATS for c in opponent_feature_cols(stat, OPP_WINDOWS)]
    return merge_opponent_features(p, defense, cols)
```

Replace `build_features` with:
```python
def build_features(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per input player-game with its features, sorted by (GAME_DATE, GAME_ID, PLAYER_ID)."""
    p = player_games[PLAYER_COLS].copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])
    t = team_games[TEAM_COLS].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])
    n_rows = len(p)

    p = _add_context(p)
    p = p.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    p = _add_rest(p)
    p = _add_rolling_minutes(p)
    p = _add_per_minute_rates(p)
    p = _add_usage(p, t)
    p = _add_team_points(p, t)
    p = _add_opponent(p, t)

    if len(p) != n_rows:
        raise ValueError(
            f"feature merges changed the row count ({n_rows} -> {len(p)}); check for duplicate team or player rows"
        )
    return p.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features_builder.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features/builder.py tests/test_features_builder.py
git commit -m "feat: complete the feature builder with team, usage and opponent features"
```

---

### Task 4: Upcoming games and the parity test

**Files:**
- Modify: `src/features/builder.py` (constants, `_upcoming_frames`, replace `build_features`)
- Test: `tests/test_features_parity.py` (create)

**Interfaces:**
- Consumes: `build_features` and the helpers from Tasks 2-3; `make_league` from `tests/league_fixture.py`.
- Produces: `build_features(player_games, team_games, upcoming_games=None, upcoming_players=None)`. `upcoming_games` columns: `GAME_ID, GAME_DATE, HOME_TEAM_ID, HOME_ABBR, AWAY_TEAM_ID, AWAY_ABBR`. `upcoming_players` columns: `GAME_ID, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION`. The result gains a boolean `IS_UPCOMING` column; upcoming rows have NaN box-score stats.

- [ ] **Step 1: Write the failing tests**

`tests/test_features_parity.py`:
```python
"""
The parity test: the features computed for a game as if it were UPCOMING (history cut off
before it, the game supplied as placeholder rows) must equal the features training computes
for that same game. This is what stops serving from drifting away from training.
"""

import pandas as pd
import pytest

from src.features.builder import build_features
from src.features.columns import BUILDER_FEATURES
from tests.league_fixture import make_league


def _split_for_serving(p, t, game_id):
    """History strictly before the game, plus the game described as upcoming."""
    date = p.loc[p["GAME_ID"] == game_id, "GAME_DATE"].iloc[0]
    team_rows = t[t["GAME_ID"] == game_id]
    is_home = team_rows["MATCHUP"].str.contains("vs.", regex=False)
    home, away = team_rows[is_home].iloc[0], team_rows[~is_home].iloc[0]
    games = pd.DataFrame(
        [{
            "GAME_ID": game_id, "GAME_DATE": date,
            "HOME_TEAM_ID": home["TEAM_ID"], "HOME_ABBR": home["TEAM_ABBREVIATION"],
            "AWAY_TEAM_ID": away["TEAM_ID"], "AWAY_ABBR": away["TEAM_ABBREVIATION"],
        }]
    )
    players = p.loc[p["GAME_ID"] == game_id, ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    return p[p["GAME_DATE"] < date], t[t["GAME_DATE"] < date], games, players


def _as_upcoming(p, t, game_id):
    return build_features(*_split_for_serving(p, t, game_id))


@pytest.mark.parametrize("game_id", ["g2", "g6", "g12"])
def test_serving_features_equal_training_features(game_id):
    p, t = make_league(n_games=12)
    training = build_features(p, t)
    serving = _as_upcoming(p, t, game_id)
    a = training[training["GAME_ID"] == game_id].sort_values("PLAYER_ID").reset_index(drop=True)[BUILDER_FEATURES]
    b = serving[serving["IS_UPCOMING"]].sort_values("PLAYER_ID").reset_index(drop=True)[BUILDER_FEATURES]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_parity_holds_across_an_offseason_gap():
    p, t = make_league(n_games=12, gap_after=6, gap_days=150)
    first_after_gap = "g7"
    training = build_features(p, t)
    serving = _as_upcoming(p, t, first_after_gap)
    a = training[training["GAME_ID"] == first_after_gap].sort_values("PLAYER_ID").reset_index(drop=True)
    b = serving[serving["IS_UPCOMING"]].sort_values("PLAYER_ID").reset_index(drop=True)
    assert (a["DAYS_REST"] == 14).all() and (b["DAYS_REST"] == 14).all()  # long rest is capped, not dropped
    assert a["MIN_ROLL_3"].notna().all()  # rolling windows carry over the gap
    pd.testing.assert_frame_equal(a[BUILDER_FEATURES], b[BUILDER_FEATURES], check_dtype=False)


def test_upcoming_rows_are_flagged_and_have_unknown_results():
    p, t = make_league()
    f = _as_upcoming(p, t, "g12")
    up = f[f["IS_UPCOMING"]]
    assert len(up) == 2
    assert up[["MIN", "PTS", "REB", "AST"]].isna().all().all()
    assert not f[~f["IS_UPCOMING"]]["MIN"].isna().any()


def test_upcoming_games_never_change_the_history_features():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    alone = build_features(hist_p, hist_t)
    combined = build_features(hist_p, hist_t, games, players)
    combined_hist = combined[~combined["IS_UPCOMING"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone[BUILDER_FEATURES], combined_hist[BUILDER_FEATURES], check_dtype=False)


def test_games_and_players_must_be_passed_together():
    p, t = make_league()
    hist_p, hist_t, games, _ = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="together"):
        build_features(hist_p, hist_t, upcoming_games=games)


def test_a_player_in_two_upcoming_games_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="more than one"):
        build_features(hist_p, hist_t, games, pd.concat([players, players.iloc[[0]]]))


def test_an_upcoming_game_already_in_history_is_rejected():
    p, t = make_league()
    _, _, games, players = _split_for_serving(p, t, "g12")
    with pytest.raises(ValueError, match="already in the history"):
        build_features(p, t, games, players)  # full history still contains g12


def test_a_player_whose_team_is_not_in_the_game_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    players = players.copy()
    players.iloc[0, players.columns.get_loc("TEAM_ABBREVIATION")] = "ZZZ"
    with pytest.raises(ValueError, match="not playing"):
        build_features(hist_p, hist_t, games, players)


def test_an_upcoming_game_dated_before_the_end_of_history_is_rejected():
    p, t = make_league()
    hist_p, hist_t, games, players = _split_for_serving(p, t, "g12")
    games = games.copy()
    games["GAME_DATE"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="before the end of the history"):
        build_features(hist_p, hist_t, games, players)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features_parity.py -v`
Expected: FAIL with `TypeError: build_features() got an unexpected keyword argument` (or takes 2 positional arguments).

- [ ] **Step 3: Implement. In `src/features/builder.py`:**

Add these constants below `TEAM_COLS`:
```python
UPCOMING_GAME_COLS = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "HOME_ABBR", "AWAY_TEAM_ID", "AWAY_ABBR"]
UPCOMING_PLAYER_COLS = ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
_PLAYER_STATS = ["MIN", "PTS", "REB", "AST", "FGA", "FTA", "FG3A", "TOV"]
_TEAM_STATS = ["PTS", "REB", "AST", "FGA", "FTA", "TOV"]
```

Add this function above `build_features`:
```python
def _upcoming_frames(games, players, history_game_ids, history_last_date):
    """Placeholder player rows and team rows (unknown box scores) for games not yet played."""
    for name, frame, cols in (("upcoming_games", games, UPCOMING_GAME_COLS), ("upcoming_players", players, UPCOMING_PLAYER_COLS)):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")

    games = games[UPCOMING_GAME_COLS].copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    if games["GAME_ID"].duplicated().any():
        raise ValueError("upcoming_games lists a game more than once")
    if set(games["GAME_ID"]) & set(history_game_ids):
        raise ValueError("an upcoming game is already in the history")
    if (games["GAME_DATE"] < history_last_date).any():
        raise ValueError("an upcoming game is dated before the end of the history")

    home = pd.DataFrame(
        {
            "GAME_ID": games["GAME_ID"], "GAME_DATE": games["GAME_DATE"],
            "TEAM_ID": games["HOME_TEAM_ID"], "TEAM_ABBREVIATION": games["HOME_ABBR"],
            "MATCHUP": games["HOME_ABBR"] + " vs. " + games["AWAY_ABBR"],
        }
    )
    away = pd.DataFrame(
        {
            "GAME_ID": games["GAME_ID"], "GAME_DATE": games["GAME_DATE"],
            "TEAM_ID": games["AWAY_TEAM_ID"], "TEAM_ABBREVIATION": games["AWAY_ABBR"],
            "MATCHUP": games["AWAY_ABBR"] + " @ " + games["HOME_ABBR"],
        }
    )
    team_rows = pd.concat([home, away], ignore_index=True)
    for col in _TEAM_STATS:
        team_rows[col] = np.nan

    pl = players[UPCOMING_PLAYER_COLS].copy()
    if pl["PLAYER_ID"].duplicated().any():
        raise ValueError("a player appears in more than one upcoming game")
    pl = pl.merge(
        team_rows[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP"]],
        on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    )
    if pl["MATCHUP"].isna().any():
        raise ValueError("an upcoming player's team is not playing in the game listed for them")
    for col in _PLAYER_STATS:
        pl[col] = np.nan
    return pl[PLAYER_COLS], team_rows[TEAM_COLS]
```

Replace `build_features` with:
```python
def build_features(
    player_games: pd.DataFrame,
    team_games: pd.DataFrame,
    upcoming_games: Optional[pd.DataFrame] = None,
    upcoming_players: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    One row per player-game, sorted by (GAME_DATE, GAME_ID, PLAYER_ID).

    player_games / team_games: the played history (see PLAYER_COLS / TEAM_COLS).
    upcoming_games / upcoming_players: optional games not yet played (see UPCOMING_*_COLS). They
    enter as placeholder rows with unknown box scores, so their features are computed by exactly
    the code that computes training features, from the history alone. Result rows for them have
    IS_UPCOMING == True and NaN stats.
    """
    if (upcoming_games is None) != (upcoming_players is None):
        raise ValueError("pass upcoming_games and upcoming_players together, or neither")

    p = player_games[PLAYER_COLS].copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])
    p["IS_UPCOMING"] = False
    t = team_games[TEAM_COLS].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])

    if upcoming_games is not None:
        up_p, up_t = _upcoming_frames(upcoming_games, upcoming_players, p["GAME_ID"], p["GAME_DATE"].max())
        up_p["IS_UPCOMING"] = True
        p = pd.concat([p, up_p], ignore_index=True)
        t = pd.concat([t, up_t], ignore_index=True)
    n_rows = len(p)

    p = _add_context(p)
    p = p.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    p = _add_rest(p)
    p = _add_rolling_minutes(p)
    p = _add_per_minute_rates(p)
    p = _add_usage(p, t)
    p = _add_team_points(p, t)
    p = _add_opponent(p, t)

    if len(p) != n_rows:
        raise ValueError(
            f"feature merges changed the row count ({n_rows} -> {len(p)}); check for duplicate team or player rows"
        )
    return p.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)
```
Also add `from typing import Optional` to the imports.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features_builder.py tests/test_features_parity.py -v`
Expected: PASS (12 builder tests + 11 parity tests).

- [ ] **Step 5: Commit**

```bash
git add src/features/builder.py tests/test_features_parity.py
git commit -m "feat: build features for upcoming games, proven equal to training by a parity test"
```

---

### Task 5: Continuous multi-season history

**Files:**
- Create: `src/features/history.py`
- Test: `tests/test_features_history.py`

**Interfaces:**
- Consumes: `PLAYER_COLS`, `TEAM_COLS`, `build_features` from `src/features/builder.py`; `validate_player_gamelog`, `validate_team_gamelog`, `DataQualityError` from `src/common/validation.py`; the existing fetchers `fetch_league_player_gamelogs` (`src/minutes/build_dataset.py`) and `fetch_league_team_gamelogs` (`src/common/opponent_stats.py`), each `(season, season_type=..., use_cache=...) -> DataFrame`.
- Produces: `load_history(seasons: Sequence[str], season_type="Regular Season", use_cache=True, fetch_players=None, fetch_teams=None) -> tuple[DataFrame, DataFrame]` returning `(player_games, team_games)` for all seasons, validated, sorted by `GAME_DATE, GAME_ID`.

- [ ] **Step 1: Write the failing tests**

`tests/test_features_history.py`:
```python
import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.features.builder import build_features
from src.features.history import load_history
from tests.league_fixture import make_league


def _season(prefix, start):
    p, t = make_league(n_games=6)
    shift = pd.Timestamp(start) - pd.Timestamp("2025-01-01")
    for frame in (p, t):
        frame["GAME_DATE"] = frame["GAME_DATE"] + shift
        frame["GAME_ID"] = prefix + frame["GAME_ID"]
    return p, t


def _fetchers(**seasons):
    def fetch_players(season, season_type="Regular Season", use_cache=True):
        return seasons[season][0]

    def fetch_teams(season, season_type="Regular Season", use_cache=True):
        return seasons[season][1]

    return fetch_players, fetch_teams


def test_seasons_are_concatenated_in_date_order():
    s1, s2 = _season("a", "2025-01-01"), _season("b", "2025-10-21")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s2})
    players, teams = load_history(["2025-26", "2024-25"], fetch_players=fp, fetch_teams=ft)
    assert len(players) == len(s1[0]) + len(s2[0])
    assert len(teams) == len(s1[1]) + len(s2[1])
    assert players["GAME_DATE"].is_monotonic_increasing


def test_the_same_game_in_two_seasons_is_rejected():
    s1 = _season("a", "2025-01-01")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s1})
    with pytest.raises(DataQualityError, match="duplicate"):
        load_history(["2024-25", "2025-26"], fetch_players=fp, fetch_teams=ft)


def test_no_seasons_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        load_history([])


def test_features_carry_across_the_season_boundary():
    s1, s2 = _season("a", "2025-01-01"), _season("b", "2025-10-21")
    fp, ft = _fetchers(**{"2024-25": s1, "2025-26": s2})
    players, teams = load_history(["2024-25", "2025-26"], fetch_players=fp, fetch_teams=ft)
    f = build_features(players, teams)
    first_of_new_season = f[(f["PLAYER_ID"] == 1) & (f["GAME_ID"] == "bg1")].iloc[0]
    assert first_of_new_season["DAYS_REST"] == 14  # the offseason, capped
    assert first_of_new_season["MIN_ROLL_3"] == pytest.approx(f[(f["PLAYER_ID"] == 1) & f["GAME_ID"].isin(["ag4", "ag5", "ag6"])]["MIN"].mean())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.history'`.

- [ ] **Step 3: Write the implementation**

`src/features/history.py`:
```python
"""
history.py: one continuous player-game log and team-game log across seasons.

Rolling windows and rest days then carry over the offseason instead of restarting every
October. Every pull is validated, and the combined history is validated again, which also
catches the same game appearing in two seasons.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

from src.common.validation import validate_player_gamelog, validate_team_gamelog
from src.features.builder import PLAYER_COLS, TEAM_COLS


def load_history(
    seasons: Sequence[str],
    season_type: str = "Regular Season",
    use_cache: bool = True,
    fetch_players: Optional[Callable] = None,
    fetch_teams: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player_games, team_games) for all `seasons`, validated and sorted by date."""
    if not seasons:
        raise ValueError("load_history needs at least one season")
    if fetch_players is None:
        from src.minutes.build_dataset import fetch_league_player_gamelogs as fetch_players
    if fetch_teams is None:
        from src.common.opponent_stats import fetch_league_team_gamelogs as fetch_teams

    players, teams = [], []
    for season in seasons:
        players.append(fetch_players(season, season_type=season_type, use_cache=use_cache)[PLAYER_COLS])
        teams.append(fetch_teams(season, season_type=season_type, use_cache=use_cache)[TEAM_COLS])

    player_games = pd.concat(players, ignore_index=True)
    team_games = pd.concat(teams, ignore_index=True)
    for frame in (player_games, team_games):
        frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])

    validate_player_gamelog(player_games, "combined player history")
    validate_team_gamelog(team_games, "combined team history")

    sort_keys = ["GAME_DATE", "GAME_ID"]
    return (
        player_games.sort_values(sort_keys).reset_index(drop=True),
        team_games.sort_values(sort_keys).reset_index(drop=True),
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features_history.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features/history.py tests/test_features_history.py
git commit -m "feat: load a continuous multi-season history, validated"
```

---

### Task 6: Trainer on the unified builder, plus the `train-all` command

**Files:**
- Create: `src/training/__init__.py`, `src/training/train_all.py`
- Modify: `main.py` (import, subcommand, dispatch)
- Test: `tests/test_train_all.py`

**Interfaces:**
- Consumes: `build_features`, `load_history`, `MINUTES_FEATURES`, `PROP_FEATURES`, `PROP_TARGETS`, `BUILDER_FEATURES`, `season_for_date`, `validate_training_frame`.
- Produces: `holdout_cutoff(df, test_ratio=0.2) -> Timestamp`; `train_models(features, season_type="Regular Season", artifacts_dir=ARTIFACTS_DIR, test_ratio=0.2, seed=42) -> dict[str, dict]` (keys `minutes, points, rebounds, assists`; each value is the eval object also written to `{name}_eval_{label}.json`); `train_all(seasons, season_type="Regular Season", use_cache=True, artifacts_dir=ARTIFACTS_DIR) -> dict`. Artifacts: `{name}_model_{label}.joblib`, `{name}_features_{label}.json`, `{name}_eval_{label}.json`, where `label = "<latest season>_<season type without spaces>"`, the filenames the current predict code reads.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_all.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

from src.common.seasons import season_for_date
from src.common.validation import DataQualityError
from src.features.columns import BUILDER_FEATURES
from src.training.train_all import holdout_cutoff, train_models


def _synthetic_features(n_per_season=700, seed=0):
    """Random but well-formed builder output spanning two seasons."""
    rng = np.random.default_rng(seed)
    frames = []
    for start in ("2024-10-22", "2025-10-21"):
        n = n_per_season
        f = pd.DataFrame(
            {
                "GAME_DATE": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 170, n), unit="D"),
                "IS_UPCOMING": False,
            }
        )
        for col in BUILDER_FEATURES:
            f[col] = rng.uniform(0.1, 30.0, n)
        f["IS_HOME"] = rng.integers(0, 2, n)
        f["IS_B2B"] = rng.integers(0, 2, n)
        f["DAYS_REST"] = rng.integers(1, 6, n)
        f["MIN"] = np.clip(0.8 * f["MIN_ROLL_5"] + rng.normal(0, 4, n), 0, 48)
        f["PTS"] = np.clip(f["MIN"] * 0.5 + rng.normal(0, 3, n), 0, None)
        f["REB"] = np.clip(f["MIN"] * 0.2 + rng.normal(0, 1.5, n), 0, None)
        f["AST"] = np.clip(f["MIN"] * 0.12 + rng.normal(0, 1.2, n), 0, None)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_holdout_is_the_last_fifth_of_the_latest_season():
    df = _synthetic_features()
    cutoff = holdout_cutoff(df, 0.2)
    assert season_for_date(cutoff) == "2025-26"
    latest = df[df["GAME_DATE"].map(season_for_date) == "2025-26"]
    share = (latest["GAME_DATE"] >= cutoff).mean()
    assert 0.15 < share < 0.25


def test_train_models_writes_artifacts_and_tests_only_on_later_dates(tmp_path):
    results = train_models(_synthetic_features(), artifacts_dir=tmp_path)

    assert set(results) == {"minutes", "points", "rebounds", "assists"}
    for name in results:
        for kind, ext in (("model", "joblib"), ("features", "json"), ("eval", "json")):
            assert (tmp_path / f"{name}_{kind}_2025-26_RegularSeason.{ext}").exists()

    ev = json.loads((tmp_path / "points_eval_2025-26_RegularSeason.json").read_text())
    assert ev["seasons"] == ["2024-25", "2025-26"]
    assert pd.Timestamp(ev["trained_through"]) < pd.Timestamp(ev["holdout_cutoff"]) <= pd.Timestamp(ev["test_date_min"])
    assert ev["rows_train"] > ev["rows_test"] > 0
    assert np.isfinite(ev["mae"]) and np.isfinite(ev["baseline_mae"])
    assert len(ev["features"]) == 26


def test_upcoming_rows_are_never_trained_on(tmp_path):
    base = _synthetic_features()
    upcoming = base.iloc[:60].copy()
    upcoming["IS_UPCOMING"] = True
    upcoming[["MIN", "PTS", "REB", "AST"]] = np.nan
    without = train_models(base, artifacts_dir=tmp_path / "a")
    with_upcoming = train_models(pd.concat([base, upcoming], ignore_index=True), artifacts_dir=tmp_path / "b")
    assert without["points"]["rows_train"] == with_upcoming["points"]["rows_train"]
    assert without["points"]["rows_test"] == with_upcoming["points"]["rows_test"]


def test_a_constant_feature_stops_training(tmp_path):
    df = _synthetic_features()
    df["DAYS_REST"] = 3  # what a source that silently went empty looks like
    with pytest.raises(DataQualityError, match="constant"):
        train_models(df, artifacts_dir=tmp_path)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_train_all.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training'`.

- [ ] **Step 3: Write the implementation**

`src/training/__init__.py`: empty file.

`src/training/train_all.py`:
```python
"""
train_all.py: train the minutes model and the three prop models on a continuous multi-season history.

Uses the same build_features() the live slate will use. The holdout is chronological: the most
recent season's last 20% of games, by date; everything earlier trains. Writes the filenames the
existing predict code reads, labelled by the latest season.

Known limitation (fixed in milestone 2): MIN_PRED is in-sample for the rows the prop models train
on, so they trust it more than they will at serving time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.seasons import season_for_date
from src.common.validation import validate_training_frame
from src.features.builder import build_features
from src.features.columns import MINUTES_FEATURES, PROP_FEATURES, PROP_TARGETS
from src.features.history import load_history

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"

MINUTES_PARAMS = {"max_depth": 6, "learning_rate": 0.08, "max_iter": 400}
PROP_PARAMS = {"max_depth": 6, "learning_rate": 0.06, "max_iter": 600}
PRED_CLIPS = {"points": (0, 80), "rebounds": (0, 40), "assists": (0, 30)}
MIN_PRED_CLIP = (0, 60)


def holdout_cutoff(df: pd.DataFrame, test_ratio: float = 0.2) -> pd.Timestamp:
    """First date of the holdout: the start of the last `test_ratio` of the LATEST season's rows."""
    seasons = df["GAME_DATE"].map(season_for_date)
    latest = df.loc[seasons == seasons.max(), "GAME_DATE"].sort_values()
    return latest.iloc[int(len(latest) * (1 - test_ratio))]


def _fit(train: pd.DataFrame, features, target: str, params: dict, seed: int):
    validate_training_frame(train, features)  # refuses missing, constant, or mostly empty features
    model = HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **params)
    model.fit(train[features].astype(float), train[target].astype(float))
    return model


def _score(y_true, y_pred, baseline) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "baseline_mae": float(mean_absolute_error(y_true, baseline)),
        "baseline_rmse": float(mean_squared_error(y_true, baseline) ** 0.5),
    }


def _save(artifacts_dir: Path, name: str, label: str, model, features, evaluation: dict) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"{name}_model_{label}.joblib")
    (artifacts_dir / f"{name}_features_{label}.json").write_text(json.dumps(list(features), indent=2))
    (artifacts_dir / f"{name}_eval_{label}.json").write_text(json.dumps(evaluation, indent=2))


def train_models(
    features: pd.DataFrame,
    season_type: str = "Regular Season",
    artifacts_dir: Path = ARTIFACTS_DIR,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, dict]:
    """Train minutes then props on build_features() output. Returns each model's eval object."""
    df = features.loc[~features["IS_UPCOMING"]].dropna(subset=["MIN", "PTS", "REB", "AST"]).copy()
    seasons = sorted(set(df["GAME_DATE"].map(season_for_date)))
    latest = seasons[-1]
    label = f"{latest}_{season_type.replace(' ', '')}"
    cutoff = holdout_cutoff(df, test_ratio)

    def meta(train: pd.DataFrame, test: pd.DataFrame) -> dict:
        return {
            "season": latest,
            "season_type": season_type,
            "seasons": seasons,
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
            "holdout_cutoff": str(cutoff.date()),
            "trained_through": str(train["GAME_DATE"].max().date()),
            "test_date_min": str(test["GAME_DATE"].min().date()),
            "test_date_max": str(test["GAME_DATE"].max().date()),
        }

    results: Dict[str, dict] = {}

    # Stage 1: minutes.
    m = df.dropna(subset=["MIN_ROLL_3", "MIN_ROLL_5", "MIN_ROLL_10"])
    m_train, m_test = m[m["GAME_DATE"] < cutoff], m[m["GAME_DATE"] >= cutoff]
    minutes_model = _fit(m_train, MINUTES_FEATURES, "MIN", MINUTES_PARAMS, seed)
    preds = np.clip(minutes_model.predict(m_test[MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP)
    evaluation = {**meta(m_train, m_test), "features": MINUTES_FEATURES, **_score(m_test["MIN"], preds, m_test["MIN_ROLL_5"])}
    _save(artifacts_dir, "minutes", label, minutes_model, MINUTES_FEATURES, evaluation)
    results["minutes"] = evaluation

    df["MIN_PRED"] = np.clip(minutes_model.predict(df[MINUTES_FEATURES].astype(float)), *MIN_PRED_CLIP)

    # Stage 2: props.
    for prop, target in PROP_TARGETS.items():
        feats = PROP_FEATURES[prop]
        needed = [c for c in feats if "ROLL" in c or c == "MIN_PRED"]
        d = df.dropna(subset=needed)
        train, test = d[d["GAME_DATE"] < cutoff], d[d["GAME_DATE"] >= cutoff]
        model = _fit(train, feats, target, PROP_PARAMS, seed)
        preds = np.clip(model.predict(test[feats].astype(float)), *PRED_CLIPS[prop])
        baseline = test["MIN_PRED"] * test[f"{target}_PER_MIN_ROLL_10"]
        evaluation = {**meta(train, test), "features": feats, **_score(test[target], preds, baseline)}
        _save(artifacts_dir, prop, label, model, feats, evaluation)
        results[prop] = evaluation

    return results


def _print_summary(results: Dict[str, dict]) -> None:
    first = next(iter(results.values()))
    print(f"\nTrained on {', '.join(first['seasons'])} through {first['trained_through']}; "
          f"holdout {first['test_date_min']} to {first['test_date_max']}")
    print(f"{'model':10s} {'MAE':>7s} {'baseline':>9s} {'lift':>7s}")
    for name, ev in results.items():
        lift = (ev["baseline_mae"] - ev["mae"]) / ev["baseline_mae"]
        print(f"{name:10s} {ev['mae']:7.3f} {ev['baseline_mae']:9.3f} {lift:+7.2%}")


def train_all(
    seasons: Sequence[str],
    season_type: str = "Regular Season",
    use_cache: bool = True,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, dict]:
    players, teams = load_history(seasons, season_type=season_type, use_cache=use_cache)
    results = train_models(build_features(players, teams), season_type=season_type, artifacts_dir=artifacts_dir)
    _print_summary(results)
    return results
```

`main.py`: add the import below the other `src` imports:
```python
from src.training.train_all import train_all
```
Add the subcommand after the `predict-prop` parser block (before `args = parser.parse_args()`):
```python
    # ---- Multi-season training (unified feature builder) ----
    p = sub.add_parser("train-all", help="Train minutes + prop models on several seasons of continuous history")
    p.add_argument("--seasons", nargs="+", required=True, help='Seasons, e.g. 2024-25 2025-26')
    p.add_argument("--season-type", type=str, default="Regular Season")
```
Add the dispatch branch at the end of the `if/elif` chain:
```python
    elif args.cmd == "train-all":
        train_all(seasons=args.seasons, season_type=args.season_type)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_train_all.py -v`
Expected: PASS (4 tests). Then `python main.py train-all --help` prints the new options.

- [ ] **Step 5: Commit**

```bash
git add src/training/__init__.py src/training/train_all.py main.py tests/test_train_all.py
git commit -m "feat: train on a continuous multi-season history with the unified builder"
```

---

### Task 7: Verify on real data, retrain on both seasons, refresh the docs

**Files:**
- Modify: `README.md` (Model Performance section), `docs/data.json` (regenerated)
- No new code. The parity script in Step 1 is a one-off check and is not committed.

**Interfaces:**
- Consumes: everything above; the old per-season datasets on disk (`data/processed/{minutes,points,rebounds,assists}_dataset_2025-26_RegularSeason.parquet`) as the reference.
- Produces: new `models_artifacts/*_2025-26_RegularSeason.*` files trained on two seasons; updated published numbers.

- [ ] **Step 1: Prove the new builder reproduces the old pipeline's features exactly**

Run from the repo root:
```bash
python - <<'EOF'
import pandas as pd
from src.features.builder import build_features
from src.features.columns import MINUTES_FEATURES, PROP_FEATURES
from src.features.history import load_history

players, teams = load_history(["2025-26"])
new = build_features(players, teams).set_index(["GAME_ID", "PLAYER_ID"])

specs = {"minutes": MINUTES_FEATURES, **{k: [c for c in v if c != "MIN_PRED"] for k, v in PROP_FEATURES.items()}}
for name, cols in specs.items():
    old = pd.read_parquet(f"data/processed/{name}_dataset_2025-26_RegularSeason.parquet").set_index(["GAME_ID", "PLAYER_ID"])
    joined = new[cols].join(old[cols], how="inner", lsuffix="_new", rsuffix="_old")
    worst = 0.0
    for c in cols:
        a, b = joined[f"{c}_new"].astype(float), joined[f"{c}_old"].astype(float)
        assert (a.isna() == b.isna()).all(), f"{name}.{c}: NaN pattern differs"
        worst = max(worst, float((a - b).abs().max(skipna=True)))
    print(f"{name:9s} rows matched {len(joined):,} of {len(old):,}; largest difference {worst:.2e}")
EOF
```
Expected: every model matches all its rows with a largest difference below `1e-6`. If any difference is larger, or a NaN pattern differs, STOP and diagnose (the likeliest cause is `TEAM_PTS_ROLL_*`, which the old code summed from player rows while the builder uses the team log). Do not loosen the check.

- [ ] **Step 2: Train on both seasons**

Run: `python main.py train-all --seasons 2024-25 2025-26`
Expected: a summary table for `minutes`, `points`, `rebounds`, `assists` with MAE, baseline MAE, and lift, for the holdout covering the last 20% of 2025-26. Record the printed numbers; Step 4 uses them. (The earlier experiment gave points +1.49% and minutes +2.11%; this run uses the new builder's `MIN_PRED`, so expect values in that neighborhood, not identical.)

- [ ] **Step 3: Confirm the old live path can still use the new models**

Run: `python main.py predict-prop --prop points --season 2025-26`
Expected: prints `Saved points predictions -> ...` with no error. (The old predict code reads the same artifact filenames and a 26-name feature list, and the old dataset has the same columns.)

- [ ] **Step 4: Refresh the published numbers**

Run: `python -m src.ui.export_static_site_data` (rewrites `docs/data.json` from the new eval files).

In `README.md`, replace the Model Performance table with a single table using the numbers printed in Step 2 (one row each for Points, Rebounds, Assists; columns `MAE`, `Baseline MAE`, `Lift`), and replace the paragraph under it with:
```
Trained on 2024-25 plus the first 80% of 2025-26 and evaluated on the last 20% of 2025-26
(a chronological split by date, never random). "Baseline" is a naive last-5-games rolling
average; "Lift" is the model's improvement over that baseline. Training on two seasons
instead of one raised the lift slightly. This is measured against a naive baseline, not
against real sportsbook lines.
```
Keep the Methodology notes section unchanged.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (the 48 existing plus the new ones from Tasks 1-6).

- [ ] **Step 6: Commit**

```bash
git add README.md docs/data.json
git commit -m "docs: publish numbers from the two-season model on the unified builder"
```

---

## Spec coverage

| Spec requirement (milestone 1) | Task |
|---|---|
| One feature builder used by training and serving | 2, 3, 4 |
| Parity test, including across an offseason | 4 |
| Continuous cross-season history | 5 |
| Retrain on both seasons, chronological holdout | 6, 7 |
| Same artifacts the current code reads (no breakage) | 6, 7 (Step 3) |
| Old per-season modules untouched until milestone 4 | Global Constraints |

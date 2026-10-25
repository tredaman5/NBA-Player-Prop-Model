# Milestone 4: Daily Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command, `python -m src.pipeline.daily`, runs the whole daily job unattended (refresh, validate, build, project, check, write) and stops loudly when a check fails. The old stale-row path is deleted.

**Architecture:** `src/pipeline/checks.py` holds the pass/fail rules that sit around `build_slate` (lines exist for the date, enough lines matched, model age). `src/pipeline/daily.py` calls the existing functions in order with injectable fetchers (so tests need no network), always writes a machine-readable run summary, and maps outcomes to exit codes. No `subprocess`, no files passed between stages.

**Tech Stack:** pandas, numpy, nba_api, joblib, pytest. No new dependencies.

**Spec:** `design/2026-10-21-slate-rewrite.md` (milestone 4 of 6).

Commands assume the project virtualenv is active and are run from the repo root. Every task is test-first: write the failing test, watch it fail, implement, watch it pass, commit.

## Global Constraints

- Feature logic stays in `src/features/builder.py`; the pipeline only orchestrates.
- No odds data leaves the machine: outputs go to `data/processed/` (gitignored), never to `docs/`.
- No "edge", "BET", EV or probability in any output. Projection, line, difference only.
- A failed check must raise (and exit non-zero), never print and continue.
- Specs and plans live in `design/`, not `docs/`.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Decisions

| Decision | Why |
|---|---|
| Fail if more than 20% of the day's lines cannot be matched to a team or player (unknown team, not on roster, ambiguous) | The 2026-10-23 replay had 0%. A jump means a provider renamed something or the roster pull broke. Thin markets and rookies are normal and do not count |
| Fail if no game on the slate date is in the lines log | The scheduled run must not publish an empty or stale slate silently. Exit code 2 so a workflow can treat "no games" differently from "broken" |
| Model age is a warning, not a failure | Retrain cadence is a separate decision. The real models were trained through 2026-03-11, so tonight's warning will be real. Threshold 60 days |
| The run summary JSON is written on success and on failure | A scheduled run needs something to read when it breaks |
| Exit codes: 0 ok, 1 failed a check, 2 no games/lines for the date | Lets M5's workflow branch without parsing text |
| `build_slate`'s own gates (stale history, roster abbreviations) stay where they are | Already tested; the pipeline just lets them propagate and records them |
| `src/odds/fetch_bets_theoddsapi.py` stays | `capture_lines.py` imports `BASE_URL`, `SPORT_KEY`, `parse_outcomes` from it |
| `edge_calculator.py`, `bankroll.py`, `snapshot_logger.py`, `export_static_site_data.py` stay untouched | Tested code; milestones 5 and 6 decide their fate |

## File Structure

- Create `src/pipeline/__init__.py` (empty), `src/pipeline/checks.py`, `src/pipeline/daily.py`.
- Create `tests/slate_fixture.py` (shared synthetic league, moved out of `tests/test_serving_slate.py`), `tests/test_pipeline_checks.py`, `tests/test_pipeline_daily.py`.
- Modify `src/serving/slate.py` (remove `main` and the CLI-only imports and constants), `tests/test_serving_slate.py` (import the shared fixture), `README.md`.
- Delete `src/ui/run_daily.py`, `src/betting/slip_from_csv.py`, `src/ui/format_betslip.py`, `src/slate/build_slate.py`, `src/slate/predict_slate.py`.

---

### Task 1: Shared test fixture and the checks

**Files:** create `tests/slate_fixture.py`, `src/pipeline/__init__.py`, `src/pipeline/checks.py`, `tests/test_pipeline_checks.py`; modify `tests/test_serving_slate.py`.

**Interfaces**
- Produces from `tests/slate_fixture.py`: constants `SLATE_DATE`, `AS_OF`, `TIP`; functions `make_models()`, `make_history()` (the 12-game BOS/DET league), `make_roster(extra=())`, `make_log(players=..., books=..., line=10.5, market="player_points", home=..., away=..., event="e1", tip=TIP)`.
- Produces from `src/pipeline/checks.py`:
  - `class NoLinesError(DataQualityError)`
  - `check_lines_for_date(log, slate_date) -> None`: raises `NoLinesError` when no event in the log tips off on `slate_date` (US Eastern).
  - `check_result(result, max_unmatched_share=0.2) -> None`: raises `NoLinesError` when `n_lines == 0`; `DataQualityError` when lines skipped for `UNMATCHED_REASONS` exceed the share of `n_lines` (checked first, so a broken mapping is reported as such), or when `n_projected == 0`.
  - `model_age_warning(meta, slate_date, max_days=60) -> Optional[str]`.
  - `UNMATCHED_REASONS = ("unknown team name", "not on either team's roster", "ambiguous name")`.

- [ ] **Step 1: Move the shared helpers into `tests/slate_fixture.py`**

Create `tests/slate_fixture.py` containing the module docstring's league description, the imports it needs, and the helpers currently at the top of `tests/test_serving_slate.py` (constants, `Stub`, `_models`, `_league`, `_roster`, `_log`), renamed to public names:

```python
"""
Shared synthetic league for slate and pipeline tests: Boston (BOS, team id 100, players 1-2) and
Detroit (DET, team id 200, players 3-4) have played 12 games, last on 2025-01-23. Tonight is
2025-01-25, BOS vs DET at 6:30pm ET. The stand-in models are simple functions of the features.
"""

import datetime as dt

import numpy as np
import pandas as pd

from src.common.teams import TEAM_ABBR
from src.features.columns import MINUTES_FEATURES, PROP_FEATURES
from src.serving.lines import LOG_COLUMNS
from src.serving.models import Models
from tests.league_fixture import make_league

SLATE_DATE = dt.date(2025, 1, 25)
AS_OF = pd.Timestamp("2025-01-25T12:00:00Z")
TIP = "2025-01-25T23:30:00+00:00"  # 6:30pm ET


class Stub:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return np.asarray(self.fn(X), dtype=float)


def make_models():
    return Models(
        minutes=Stub(lambda X: X["MIN_ROLL_5"]),
        props={"points": Stub(lambda X: X["MIN_PRED"] * 0.5),
               "rebounds": Stub(lambda X: X["MIN_PRED"] * 0.2),
               "assists": Stub(lambda X: X["MIN_PRED"] * 0.1)},
        features={"minutes": MINUTES_FEATURES, **PROP_FEATURES},
        rmse={"points": 5.0, "rebounds": 2.0, "assists": 1.5},
        trained_through="2025-01-01",
        label="test",
    )


def make_history():
    return make_league(n_games=12, teams=("BOS", "DET"))


def make_roster(extra=()):
    rows = [(1, "P1", 100, "BOS"), (2, "P2", 100, "BOS"), (3, "P3", 200, "DET"), (4, "P4", 200, "DET"), *extra]
    others = [t for t in sorted(set(TEAM_ABBR.values())) if t not in ("BOS", "DET")]
    rows += [(1000 + i, f"Filler{i}", 300 + i, t) for i, t in enumerate(others)]
    return pd.DataFrame(rows, columns=["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_ABBREVIATION"])
```

Then append `make_log` using the body of the existing `_log` helper verbatim (copy it from `tests/test_serving_slate.py`, the function starting `def _log(players=("P1", "P2", "P3", "P4"), books=("dk", "fd", "mgm"), line=10.5, market="player_points",`), renamed `make_log`.

In `tests/test_serving_slate.py`, delete the moved constants, `Stub`, `_models`, `_league`, `_roster`, `_log` and replace them with:

```python
from tests.slate_fixture import (
    AS_OF, SLATE_DATE, TIP,
    make_history as _league, make_log as _log, make_models as _models, make_roster as _roster,
)
```

Remove imports in that file that become unused (`numpy`, `Models`, `LOG_COLUMNS`, `TEAM_ABBR`, `MINUTES_FEATURES`, `PROP_FEATURES`, `make_league`) only if nothing else in the file uses them (grep first).

- [ ] **Step 2: Run the slate tests to confirm the refactor changed nothing**

Run: `python -m pytest tests/test_serving_slate.py -q`
Expected: 18 passed.

- [ ] **Step 3: Commit the refactor**

```bash
git add tests/slate_fixture.py tests/test_serving_slate.py
git commit -m "test: share the synthetic slate league between slate and pipeline tests"
```

- [ ] **Step 4: Write the failing tests for the checks**

`tests/test_pipeline_checks.py`:

```python
import datetime as dt

import pandas as pd
import pytest

from src.common.validation import DataQualityError
from src.pipeline.checks import (
    NoLinesError, check_lines_for_date, check_result, model_age_warning,
)
from src.serving.slate import SlateResult
from tests.slate_fixture import SLATE_DATE, make_log


def _result(n_lines, n_projected, reasons=()):
    skipped = pd.DataFrame({
        "player": [f"p{i}" for i in range(len(reasons))], "event_id": "e1",
        "market": "player_points", "reason": list(reasons),
    })
    meta = {"n_lines": n_lines, "n_projected": n_projected, "n_skipped": len(reasons),
            "model_trained_through": "2025-01-01"}
    return SlateResult(pd.DataFrame(), skipped, meta)


def test_a_log_with_a_game_on_the_date_passes():
    check_lines_for_date(make_log(), SLATE_DATE)


def test_a_log_with_no_game_on_the_date_raises_no_lines():
    with pytest.raises(NoLinesError, match="no games"):
        check_lines_for_date(make_log(), dt.date(2025, 1, 26))


def test_no_lines_is_still_a_data_quality_error():
    assert issubclass(NoLinesError, DataQualityError)


def test_a_healthy_result_passes_even_with_thin_and_rookie_skips():
    check_result(_result(100, 85, ["line from only 2 book(s)"] * 8 + ["only 1 prior games"] * 7))


def test_many_unmatched_names_fail_the_run():
    with pytest.raises(DataQualityError, match="matched"):
        check_result(_result(100, 60, ["not on either team's roster"] * 40))


def test_unknown_team_names_count_as_unmatched():
    with pytest.raises(DataQualityError, match="matched"):
        check_result(_result(10, 5, ["unknown team name"] * 5))


def test_nothing_projected_fails_the_run():
    with pytest.raises(DataQualityError, match="0 of 5"):
        check_result(_result(5, 0, ["only 1 prior games"] * 5))


def test_no_lines_left_raises_no_lines():
    with pytest.raises(NoLinesError, match="started"):
        check_result(_result(0, 0))


def test_old_models_give_a_warning_and_fresh_ones_do_not():
    old = {"model_trained_through": "2024-10-01"}
    assert "retrain" in model_age_warning(old, SLATE_DATE)
    assert model_age_warning({"model_trained_through": "2025-01-01"}, SLATE_DATE) is None
```

- [ ] **Step 5: Run to verify it fails**

Run: `python -m pytest tests/test_pipeline_checks.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.pipeline'`.

- [ ] **Step 6: Implement**

`src/pipeline/__init__.py` is empty. `src/pipeline/checks.py`:

```python
"""
checks.py: pass/fail rules around the slate build.

build_slate already refuses a stale history and a roster missing a team. These are the rules that
need the whole run's numbers: are there games to project, did the names still match, how old are the
models. A failed rule raises; the daily command turns that into a non-zero exit.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from src.common.validation import DataQualityError
from src.serving.slate import EASTERN, SlateResult

UNMATCHED_REASONS = ("unknown team name", "not on either team's roster", "ambiguous name")
MAX_UNMATCHED_SHARE = 0.2
MAX_MODEL_AGE_DAYS = 60


class NoLinesError(DataQualityError):
    """There is nothing to project for the date: no games in the log, or every game has started."""


def check_lines_for_date(log: pd.DataFrame, slate_date: dt.date) -> None:
    tips = log["commence_time"].dt.tz_convert(EASTERN).dt.date
    if not (tips == slate_date).any():
        raise NoLinesError(
            f"the lines log has no games on {slate_date} (no games that day, or the capture missed them)"
        )


def check_result(result: SlateResult, max_unmatched_share: float = MAX_UNMATCHED_SHARE) -> None:
    meta = result.meta
    n_lines = meta["n_lines"]
    if n_lines == 0:
        raise NoLinesError("no lines left to project: every captured game for that date has already started")
    unmatched = int(result.skipped["reason"].isin(UNMATCHED_REASONS).sum())
    share = unmatched / n_lines
    if share > max_unmatched_share:
        raise DataQualityError(
            f"{unmatched} of {n_lines} lines ({share:.0%}) could not be matched to a team or player; "
            "a provider name or roster change may have broken the mapping"
        )
    if meta["n_projected"] == 0:
        raise DataQualityError(f"0 of {n_lines} lines could be projected; see the skipped list for why")


def model_age_warning(meta: dict, slate_date: dt.date, max_days: int = MAX_MODEL_AGE_DAYS) -> Optional[str]:
    trained = pd.Timestamp(meta["model_trained_through"]).date()
    age = (slate_date - trained).days
    if age > max_days:
        return f"models were trained through {trained}, {age} days before this slate; retrain"
    return None
```

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest tests/test_pipeline_checks.py tests/test_serving_slate.py -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline tests/test_pipeline_checks.py
git commit -m "feat(pipeline): checks for missing games, broken name matching and old models"
```

### Task 2: The daily pipeline command

**Files:** create `src/pipeline/daily.py`, `tests/test_pipeline_daily.py`; modify `src/serving/slate.py` (remove `main`, CLI-only imports, `OUT_DIR`).

**Interfaces**
- Consumes: `build_slate`, `render_slate`, `SlateResult` (`src/serving/slate.py`); `load_line_log`; `load_models`; `fetch_roster`; `load_history`; `season_for_date`, `season_label`; the Task 1 checks.
- Produces:
  - `run_pipeline(slate_date, log_path, out_dir=OUT_DIR, include_started=False, as_of=None, load_history_fn=None, fetch_roster_fn=None, load_models_fn=None) -> (dict, SlateResult)`. Writes `slate_<date>.csv` (success only) and `run_<date>.json` (always). On a `DataQualityError` it writes the failed summary and re-raises.
  - The summary dict keys: `status` ("ok" or "failed"), `slate_date`, `as_of`, `warnings` (list), and on success also `history_last_date`, `model_label`, `model_trained_through`, `n_lines`, `n_projected`, `n_skipped`, `skipped_by_reason` (digits in reasons replaced by `N`). On failure also `error`.
  - `main(argv=None) -> int`: flags `--date`, `--include-started`, `--log-path`, `--out-dir`; log path defaults to `LINES_LOG_PATH`; exit 0 ok, 2 for `NoLinesError`, 1 for any other `DataQualityError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline_daily.py`:

```python
import dataclasses
import json

import pandas as pd
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
    on_disk = json.loads((tmp_path / "out" / "run_2025-01-25.json").read_text())
    assert on_disk["status"] == "ok" and on_disk["history_last_date"] == "2025-01-23"
    assert len(result.slate) == 4


def test_history_covers_this_and_last_season_and_refreshes_only_this_one(tmp_path):
    calls = []
    _run(tmp_path, history_calls=calls)
    assert calls == [(["2023-24", "2024-25"], ["2024-25"])]


def test_skipped_reasons_are_counted_with_the_numbers_masked(tmp_path):
    summary, _ = _run(tmp_path, log=make_log(players=("P1", "P2", "P3", "P4", "P5"), books=("dk", "fd", "mgm")))
    assert summary["skipped_by_reason"] == {"not on either team's roster": 1}


def test_a_failed_check_writes_a_failed_summary_and_raises(tmp_path):
    with pytest.raises(DataQualityError, match="matched"):
        _run(tmp_path, log=make_log(players=("Nobody A", "Nobody B")))
    on_disk = json.loads((tmp_path / "out" / "run_2025-01-25.json").read_text())
    assert on_disk["status"] == "failed" and "matched" in on_disk["error"]
    assert not (tmp_path / "out" / "slate_2025-01-25.csv").exists()


def test_a_date_with_no_games_raises_no_lines(tmp_path):
    import datetime as dt

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
```

Player `P5` in the skipped-reasons test is not on either roster, so one line is skipped for `not on either team's roster` out of 5 (20%, at the limit, so it passes: the rule fails only when the share is strictly greater than 0.2).

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pipeline_daily.py -q`
Expected: FAIL, `ImportError: cannot import name 'daily'`.

- [ ] **Step 3: Implement `src/pipeline/daily.py`**

```python
"""
daily.py: the whole daily job in one command.

    python -m src.pipeline.daily [--date YYYY-MM-DD] [--include-started] [--log-path PATH] [--out-dir DIR]

Refresh the current season's history, load rosters and the newest models, build the slate, run the
checks, then write data/processed/slate_<date>.csv and run_<date>.json. A failed check raises and
exits non-zero (1 for a broken input, 2 when there are simply no games or lines for the date); the
summary file is written either way so a scheduled run has something to read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

from src.common.seasons import season_for_date, season_label
from src.common.validation import DataQualityError
from src.pipeline.checks import NoLinesError, check_lines_for_date, check_result, model_age_warning
from src.serving.lines import load_line_log
from src.serving.slate import EASTERN, SlateResult, build_slate, render_slate

OUT_DIR = Path("data/processed")


def _write_summary(out_dir: Path, slate_date: dt.date, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run_{slate_date}.json").write_text(json.dumps(summary, indent=2, default=str))


def run_pipeline(
    slate_date: dt.date,
    log_path,
    out_dir: Path = OUT_DIR,
    include_started: bool = False,
    as_of: Optional[pd.Timestamp] = None,
    load_history_fn: Optional[Callable] = None,
    fetch_roster_fn: Optional[Callable] = None,
    load_models_fn: Optional[Callable] = None,
) -> Tuple[dict, SlateResult]:
    from src.features.history import load_history
    from src.serving.models import load_models
    from src.serving.rosters import fetch_roster

    load_history_fn = load_history_fn or load_history
    fetch_roster_fn = fetch_roster_fn or fetch_roster
    load_models_fn = load_models_fn or load_models
    out_dir = Path(out_dir)
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    summary = {"slate_date": str(slate_date), "as_of": str(as_of), "warnings": []}

    try:
        log = load_line_log(log_path)
        check_lines_for_date(log, slate_date)

        season = season_for_date(slate_date)
        seasons = [season_label(int(season[:4]) - 1), season]
        players, teams = load_history_fn(seasons, refresh_seasons=[season])
        result = build_slate(
            log, fetch_roster_fn(season), players, teams, load_models_fn(),
            as_of=as_of, slate_date=slate_date, include_started=include_started,
        )
        check_result(result)
    except DataQualityError as exc:
        summary.update(status="failed", error=str(exc))
        _write_summary(out_dir, slate_date, summary)
        raise

    warning = model_age_warning(result.meta, slate_date)
    if warning:
        summary["warnings"].append(warning)
    reasons = result.skipped["reason"].map(lambda r: re.sub(r"\d+", "N", r))
    summary.update(
        status="ok",
        history_last_date=result.meta["history_last_date"],
        model_label=result.meta["model_label"],
        model_trained_through=result.meta["model_trained_through"],
        n_lines=result.meta["n_lines"],
        n_projected=result.meta["n_projected"],
        n_skipped=result.meta["n_skipped"],
        skipped_by_reason={k: int(v) for k, v in reasons.value_counts().items()},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.slate.to_csv(out_dir / f"slate_{slate_date}.csv", index=False)
    _write_summary(out_dir, slate_date, summary)
    return summary, result


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.common.env import load_env_file

    ap = argparse.ArgumentParser(description="Run the daily slate pipeline.")
    ap.add_argument("--date", help="slate date, YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--include-started", action="store_true", help="keep games that already tipped off")
    ap.add_argument("--log-path", help="captured lines CSV (default: LINES_LOG_PATH from the environment or .env)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="where the slate CSV and run summary go")
    args = ap.parse_args(argv)

    load_env_file()
    log_path = args.log_path or os.getenv("LINES_LOG_PATH")
    if not log_path:
        ap.error("no lines log: pass --log-path or set LINES_LOG_PATH in .env")
    slate_date = (
        dt.date.fromisoformat(args.date) if args.date
        else pd.Timestamp.now(tz="UTC").tz_convert(EASTERN).date()
    )

    try:
        summary, result = run_pipeline(
            slate_date, log_path, out_dir=Path(args.out_dir), include_started=args.include_started
        )
    except NoLinesError as exc:
        print(f"NO SLATE: {exc}", file=sys.stderr)
        return 2
    except DataQualityError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(render_slate(result))
    for warning in summary["warnings"]:
        print(f"\nWARNING: {warning}")
    print(f"\nWrote {Path(args.out_dir) / f'slate_{slate_date}.csv'} and run_{slate_date}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Then in `src/serving/slate.py` delete `main`, the `if __name__` block, `OUT_DIR`, and the imports only `main` used (`argparse`, `os`, `Path`, `Optional`, `Sequence`; keep any still used elsewhere in the file; grep each).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pipeline_daily.py tests/test_serving_slate.py tests/test_pipeline_checks.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/daily.py src/serving/slate.py tests/test_pipeline_daily.py
git commit -m "feat(pipeline): one daily command with checks, a run summary and exit codes"
```

### Task 3: Retire the old path and update the docs

**Files:** delete `src/ui/run_daily.py`, `src/betting/slip_from_csv.py`, `src/ui/format_betslip.py`, `src/slate/build_slate.py`, `src/slate/predict_slate.py`; modify `README.md`.

- [ ] **Step 1: Confirm nothing live imports them**

Run: `grep -rn "run_daily\|slip_from_csv\|format_betslip\|src.slate" src tests main.py .github`
Expected: only matches inside the files being deleted, plus comment mentions in `src/backtests/snapshot_logger.py` (reword those comments to say the betslip CSV is the old format, and that milestone 6 replaces this input). No code imports.

- [ ] **Step 2: Delete and run the whole suite**

```bash
git rm src/ui/run_daily.py src/betting/slip_from_csv.py src/ui/format_betslip.py src/slate/build_slate.py src/slate/predict_slate.py
python -m pytest tests -q
python -c "import src.ui.export_static_site_data, src.odds.capture_lines, src.backtests.snapshot_logger"
```
Expected: all tests pass; the import line prints nothing.

- [ ] **Step 3: Update the README**

- Replace the "Daily Run (One Command)" section with the daily pipeline: the command, what it checks (games in the log for the date, 20% unmatched limit, 3-day history staleness, roster team codes), the exit codes (0, 1, 2), the two files it writes, and the model-age warning. Drop the "known limitation" paragraph (the stale-row builder is gone).
- In the "Live Slate" section, change the command to `python -m src.pipeline.daily` (keep `--date` and `--include-started` examples) and mention `run_<date>.json`.
- In the project structure block, remove the `slate/` line and add `pipeline/` ("daily command and its checks"); update the "Edge Calculation", "What This Project Does" and "Prediction Flow" parts that still describe the old probability/Kelly flow so they say plainly: the pipeline outputs projection, line and difference; the edge/EV/Kelly math exists in `src/betting/edge_calculator.py` but is not run or shown until the backtest supports it.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: retire run_daily and the stale-row slate path; document the daily pipeline"
```

### Task 4: Real-data verification and finish

No new code.

- [ ] **Step 1:** Run `python -m src.pipeline.daily --log-path <private log>` for today and `--date 2026-10-23 --include-started` for a replay. Check the slate prints, the run JSON has the counts, and the model-age warning appears (models trained through 2026-03-11).
- [ ] **Step 2:** Prove the failure paths on real data: `--date 2026-10-31` should exit 2 with "no games"; a copy of the log with team names mangled (in the scratchpad, never the data repo) should exit 1 with the unmatched message. Confirm `echo $?` for each.
- [ ] **Step 3:** Run the full suite. Run `git status` and confirm nothing under `data/` or the private log is staged.
- [ ] **Step 4:** Use superpowers:finishing-a-development-branch with the user (merge / PR / keep). Push only on "push".

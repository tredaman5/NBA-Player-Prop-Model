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
    models_dir=None,
) -> Tuple[dict, SlateResult]:
    from src.features.history import load_history
    from src.serving.models import load_models
    from src.serving.rosters import fetch_roster

    load_history_fn = load_history_fn or load_history
    fetch_roster_fn = fetch_roster_fn or fetch_roster
    if load_models_fn is None:
        load_models_fn = (lambda: load_models(artifacts_dir=Path(models_dir))) if models_dir else load_models
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
    ap.add_argument("--models-dir", help="folder holding the trained models (default: models_artifacts)")
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
            slate_date, log_path, out_dir=Path(args.out_dir), include_started=args.include_started,
            models_dir=args.models_dir,
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

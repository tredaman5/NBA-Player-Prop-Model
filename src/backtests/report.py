"""
report.py: the backtest command.

    python -m src.backtests.report [--log-path PATH] [--models-dir DIR] [--out-dir DIR]

Replays every captured line against the box scores and prints what the pre-registered rules
(design/backtest-rules.md) say about it. Until enough lines exist the report is interim and says so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from src.backtests.metrics import summarize
from src.backtests.replay import replay_lines
from src.backtests.rules import FIRST_GAME_DATE, VERDICT_MIN_LINES
from src.serving.models import Models

OUT_DIR = Path("data/processed")


def evaluate(log, players, teams, models: Models, first_game_date=FIRST_GAME_DATE) -> Tuple[pd.DataFrame, dict]:
    replayed = replay_lines(log, players, teams, models, first_game_date=first_game_date)
    return replayed, summarize(replayed)


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def render_report(summary: dict, dry_run: bool = False) -> str:
    s = summary
    out = []
    if dry_run:
        out += ["DRY RUN: this includes games outside the pre-registered sample. Not a result.", ""]
    out.append(f"Backtest for model {s['model_label']}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(s["status_counts"].items(), key=lambda kv: -kv[1]))
    out.append(f"All captured lines by status: {counts or 'none'}")
    if s["n_directional"] == 0:
        out += ["", "No scored lines yet. Lines count once their game is played and falls inside the sample."]
        return "\n".join(out)

    out.append(f"Scored games: {s['first_game']} to {s['last_game']}; "
               f"{s['n_scored']} lines scored, {s['n_directional']} with a direction (no pushes).")
    out.append("")
    if s["verdict"] == "interim":
        out.append(f"INTERIM, not a verdict: {s['n_directional']:,} of {VERDICT_MIN_LINES:,} lines needed "
                   f"({s['lines_needed']:,} to go).")
    elif s["verdict"] == "edge_shown":
        out.append(f"RESULT: edge shown for model {s['model_label']} under the pre-registered rules.")
    else:
        out.append(f"RESULT: no edge shown for model {s['model_label']} under the pre-registered rules.")
    lo, hi = s["hit_ci"]
    out.append(f"The model's side landed on {_pct(s['hit_rate'])} of lines (95% range {_pct(lo)} to {_pct(hi)}); "
               f"break-even is {_pct(0.524)} (about {_pct(s['breakeven_mean'])} at the prices posted).")
    lo, hi = s["d_ci"]
    out.append(f"The projection was closer than the line on {_pct(s['model_closer_share'])} of lines; "
               f"mean gain {s['mean_d']:+.2f} points (95% range {lo:+.2f} to {hi:+.2f}). "
               f"Average miss: model {s['mae_model']:.2f}, line {s['mae_market']:.2f}.")
    out += ["", "Exploratory, not part of the verdict (by size of the gap):"]
    for b in s["buckets"]:
        top = "+" if b["gap_to"] is None else f"-{b['gap_to']}"
        rate = "n/a" if b["hit_rate"] is None else _pct(b["hit_rate"])
        out.append(f"  gap {b['gap_from']}{top}: {b['n']} lines, side landed {rate}")
    out.append(f"Lines were captured about {s['mean_mins_before_tip']:.0f} minutes before tip-off on average.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.common.env import load_env_file
    from src.common.seasons import season_for_date, season_label
    from src.features.history import load_history
    from src.serving.lines import load_line_log
    from src.serving.models import load_models

    ap = argparse.ArgumentParser(description="Score captured lines against what happened.")
    ap.add_argument("--log-path", help="captured lines CSV (default: LINES_LOG_PATH from the environment or .env)")
    ap.add_argument("--models-dir", help="folder holding the trained models (default: models_artifacts)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--first-game-date", help="DRY RUN only: count games from this date instead of the "
                                              "pre-registered start. Never report the output as a result.")
    args = ap.parse_args(argv)

    load_env_file()
    log_path = args.log_path or os.getenv("LINES_LOG_PATH")
    if not log_path:
        ap.error("no lines log: pass --log-path or set LINES_LOG_PATH in .env")
    dry = args.first_game_date is not None
    first = dt.date.fromisoformat(args.first_game_date) if dry else FIRST_GAME_DATE

    season = season_for_date(pd.Timestamp.now())
    seasons = [season_label(int(season[:4]) - 1), season]
    players, teams = load_history(seasons, refresh_seasons=[season])
    models = load_models(artifacts_dir=Path(args.models_dir)) if args.models_dir else load_models()
    replayed, summary = evaluate(load_line_log(log_path), players, teams, models, first_game_date=first)
    print(render_report(summary, dry_run=dry))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{summary['model_label']}" + ("_dryrun" if dry else "")
    replayed.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {stem}.csv and .json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

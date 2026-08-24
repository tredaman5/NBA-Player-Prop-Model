from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path


def run(cmd: str) -> None:
    print(f"\n> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (UTC date). Default: today UTC.")
    ap.add_argument("--no-print", action="store_true", help="Skip pretty CLI output.")
    args = ap.parse_args()

    date_str = args.date
    if not date_str:
        date_str = dt.datetime.now(dt.timezone.utc).date().isoformat()

    # 1) fetch props odds -> bets_today.csv
    run("python -m src.odds.fetch_bets_theoddsapi")

    # 2) build slate features for date
    run(f"python -m src.slate.build_slate --date {date_str} --prop all")

    # 3) predict slate for date
    run(f"python -m src.slate.predict_slate --date {date_str} --prop all")

    # 4) generate betslip results
    run("python -m src.betting.slip_from_csv --in bets_today.csv")

    # 5) log today's predictions to the season-long backtest log, and
    #    backfill actual results for previously-logged games that have
    #    since completed (safe to re-run daily; only fills in blanks)
    run(
        "python -c \"from src.backtests.snapshot_logger import append_daily_snapshot; "
        f"append_daily_snapshot('data/processed/betslip_results.csv', log_date='{date_str}')\""
    )
    run(
        "python -c \"from src.backtests.snapshot_logger import backfill_results; backfill_results()\""
    )

    # 6) pretty print
    if not args.no_print:
        run("python -c \"from src.ui.format_betslip import format_betslip; format_betslip('data/processed/betslip_results.csv')\"")


if __name__ == "__main__":
    main()

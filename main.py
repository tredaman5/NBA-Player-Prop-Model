"""
main.py

One command runner for:
- building minutes dataset
- training minutes model
- predicting minutes
- building prop datasets (points/rebounds/assists)
- training prop models
- predicting props

Examples:
  python main.py build-minutes --season-start 2025
  python main.py train-minutes --season 2025-26
  python main.py predict-minutes --season 2025-26

  python main.py build-prop --prop points --season 2025-26
  python main.py train-prop --prop rebounds --season 2025-26
  python main.py predict-prop --prop assists --season 2025-26
"""

from __future__ import annotations

import argparse

from src.minutes.build_dataset import build_minutes_dataset
from src.minutes.train import train_minutes_model
from src.minutes.predict import predict_minutes

from src.props.points.build_dataset import build_points_dataset
from src.props.points.train import train_points_model
from src.props.points.predict import predict_points

from src.props.rebounds.build_dataset import build_rebounds_dataset
from src.props.rebounds.train import train_rebounds_model
from src.props.rebounds.predict import predict_rebounds

from src.props.assists.build_dataset import build_assists_dataset
from src.props.assists.train import train_assists_model
from src.props.assists.predict import predict_assists


def main() -> None:
    parser = argparse.ArgumentParser(prog="nba-props-ml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- Minutes ----
    p = sub.add_parser("build-minutes", help="Build minutes dataset (league-wide)")
    p.add_argument("--season-start", type=int, required=True, help="Season start year, e.g. 2025 -> 2025-26")
    p.add_argument("--season-type", type=str, default="Regular Season")
    p.add_argument("--cutoff-date", type=str, default=None, help="YYYY-MM-DD optional cutoff for in-progress season")
    p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("train-minutes", help="Train minutes model")
    p.add_argument("--season", type=str, required=True, help='Season string like "2025-26"')
    p.add_argument("--season-type", type=str, default="Regular Season")

    p = sub.add_parser("predict-minutes", help="Predict minutes using trained model")
    p.add_argument("--season", type=str, required=True)
    p.add_argument("--season-type", type=str, default="Regular Season")

    # ---- Props ----
    p = sub.add_parser("build-prop", help="Build prop dataset (points/rebounds/assists)")
    p.add_argument("--prop", choices=["points", "rebounds", "assists"], required=True)
    p.add_argument("--season", type=str, required=True)
    p.add_argument("--season-type", type=str, default="Regular Season")

    p = sub.add_parser("train-prop", help="Train prop model (points/rebounds/assists)")
    p.add_argument("--prop", choices=["points", "rebounds", "assists"], required=True)
    p.add_argument("--season", type=str, required=True)
    p.add_argument("--season-type", type=str, default="Regular Season")

    p = sub.add_parser("predict-prop", help="Predict prop values (points/rebounds/assists)")
    p.add_argument("--prop", choices=["points", "rebounds", "assists"], required=True)
    p.add_argument("--season", type=str, required=True)
    p.add_argument("--season-type", type=str, default="Regular Season")

    args = parser.parse_args()

    if args.cmd == "build-minutes":
        build_minutes_dataset(
            season_start_year=args.season_start,
            season_type=args.season_type,
            cutoff_date=args.cutoff_date,
            use_cache=not args.no_cache,
        )

    elif args.cmd == "train-minutes":
        train_minutes_model(season=args.season, season_type=args.season_type)

    elif args.cmd == "predict-minutes":
        predict_minutes(season=args.season, season_type=args.season_type)

    elif args.cmd == "build-prop":
        if args.prop == "points":
            build_points_dataset(season=args.season, season_type=args.season_type)
        elif args.prop == "rebounds":
            build_rebounds_dataset(season=args.season, season_type=args.season_type)
        else:
            build_assists_dataset(season=args.season, season_type=args.season_type)

    elif args.cmd == "train-prop":
        if args.prop == "points":
            train_points_model(season=args.season, season_type=args.season_type)
        elif args.prop == "rebounds":
            train_rebounds_model(season=args.season, season_type=args.season_type)
        else:
            train_assists_model(season=args.season, season_type=args.season_type)

    elif args.cmd == "predict-prop":
        if args.prop == "points":
            predict_points(season=args.season, season_type=args.season_type)
        elif args.prop == "rebounds":
            predict_rebounds(season=args.season, season_type=args.season_type)
        else:
            predict_assists(season=args.season, season_type=args.season_type)


if __name__ == "__main__":
    main()

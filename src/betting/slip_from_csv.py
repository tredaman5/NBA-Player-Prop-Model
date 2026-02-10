"""
slip_from_csv.py

Reads a CSV of candidate bets (player_id + game_id + prop + line + odds)
and joins against your *_with_pred parquet files to compute:

- model probability P(over line)
- implied probability from odds
- edge, EV, fractional Kelly sizing
- BET / NO BET decision

Input CSV columns (required):
  prop, season, season_type, player_id, game_id, line, odds_over
Optional:
  odds_under, kelly_multiplier, min_edge

Example:
  points,2025-26,Regular Season,2544,0022500001,24.5,-110,-110,0.25,0.02
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.betting.edge_calculator import compute_edge, should_bet


def repo_root() -> Path:
    # src/betting/file.py -> parents[2] = repo
    return Path(__file__).resolve().parents[2]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _p_over_from_mu_sigma(mu: float, sigma: float, line: float) -> float:
    sigma = float(sigma)
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return float(1.0 - _normal_cdf(z))


def _paths_for_prop(prop: str, season: str, season_type: str, root: Path) -> tuple[Path, Path]:
    st = season_type.replace(" ", "")
    processed = root / "data" / "processed"
    artifacts = root / "models_artifacts"

    if prop == "points":
        pred_path = processed / f"points_with_pred_{season}_{st}.parquet"
        eval_path = artifacts / f"points_eval_{season}_{st}.json"
    elif prop == "rebounds":
        pred_path = processed / f"rebounds_with_pred_{season}_{st}.parquet"
        eval_path = artifacts / f"rebounds_eval_{season}_{st}.json"
    elif prop == "assists":
        pred_path = processed / f"assists_with_pred_{season}_{st}.parquet"
        eval_path = artifacts / f"assists_eval_{season}_{st}.json"
    else:
        raise ValueError(f"Unknown prop: {prop}")

    return pred_path, eval_path


def _mu_col(prop: str) -> str:
    return {"points": "PTS_PRED", "rebounds": "REB_PRED", "assists": "AST_PRED"}[prop]


def run_slip(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    root = repo_root()
    bets = pd.read_csv(input_csv)

    required = ["prop", "season", "season_type", "player_id", "game_id", "line", "odds_over"]
    missing = [c for c in required if c not in bets.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Defaults
    if "odds_under" not in bets.columns:
        bets["odds_under"] = np.nan
    if "kelly_multiplier" not in bets.columns:
        bets["kelly_multiplier"] = 0.25
    if "min_edge" not in bets.columns:
        bets["min_edge"] = 0.02

    results = []

    # Cache loaded prediction tables + sigmas by (prop, season, season_type)
    pred_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    sigma_cache: dict[tuple[str, str, str], float] = {}

    for i, row in bets.iterrows():
        prop = str(row["prop"]).strip().lower()
        season = str(row["season"]).strip()
        season_type = str(row["season_type"]).strip()
        player_id = int(row["player_id"])
        game_id = str(row["game_id"]).strip()
        line = float(row["line"])
        odds_over = int(row["odds_over"])
        odds_under = row.get("odds_under", np.nan)
        odds_under = None if pd.isna(odds_under) else int(odds_under)
        k_mult = float(row.get("kelly_multiplier", 0.25))
        min_edge = float(row.get("min_edge", 0.02))

        key = (prop, season, season_type)

        if key not in pred_cache:
            pred_path, eval_path = _paths_for_prop(prop, season, season_type, root)
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"Missing predictions parquet: {pred_path}\n"
                    f"Run: python main.py predict-prop --prop {prop} --season {season}"
                )
            dfp = pd.read_parquet(pred_path)
            pred_cache[key] = dfp

            # sigma from eval RMSE (fallback per prop)
            fallback = {"points": 4.0, "rebounds": 2.0, "assists": 1.7}[prop]
            sigma = fallback
            if eval_path.exists():
                ev = json.loads(eval_path.read_text())
                sigma = float(ev.get("rmse", fallback))
            sigma_cache[key] = sigma

        dfp = pred_cache[key]
        sigma = sigma_cache[key]
        mu_col = _mu_col(prop)

        match = dfp[(dfp["PLAYER_ID"] == player_id) & (dfp["GAME_ID"].astype(str) == game_id)]
        if match.empty:
            results.append(
                {
                    "prop": prop,
                    "season": season,
                    "season_type": season_type,
                    "player_id": player_id,
                    "game_id": game_id,
                    "line": line,
                    "error": "No matching row in predictions parquet (check player_id/game_id).",
                }
            )
            continue

        mu = float(match.iloc[0][mu_col])
        p_over = _p_over_from_mu_sigma(mu=mu, sigma=sigma, line=line)

        # Evaluate both sides if odds_under provided; else only over.
        over_res = compute_edge(p_over=p_over, odds_over=odds_over, odds_under=odds_under, side="over", kelly_multiplier=k_mult)
        under_res = None
        if odds_under is not None:
            under_res = compute_edge(p_over=p_over, odds_over=odds_over, odds_under=odds_under, side="under", kelly_multiplier=k_mult)

        # Choose best EV side (or just over)
        chosen = over_res
        if under_res is not None and under_res.ev_per_dollar > over_res.ev_per_dollar:
            chosen = under_res

        decision = "BET" if should_bet(chosen, min_edge=min_edge, min_ev=0.0) else "NO BET"

        results.append(
            {
                "prop": prop,
                "season": season,
                "season_type": season_type,
                "player_id": player_id,
                "game_id": game_id,
                "line": line,
                "sigma_used": sigma,
                "p_over": p_over,
                "odds_over": odds_over,
                "odds_under": odds_under,
                "chosen_side": chosen.side,
                "p_model_side": chosen.p_model,
                "p_implied_side": chosen.p_implied,
                "edge": chosen.edge,
                "ev_per_dollar": chosen.ev_per_dollar,
                "full_kelly": chosen.kelly_fraction,
                "fractional_kelly": chosen.suggested_fractional_kelly,
                "decision": decision,
            }
        )

    out = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved betslip results -> {output_csv}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV of bets")
    ap.add_argument("--out", dest="out", default="data/processed/betslip_results.csv", help="Output CSV path")
    args = ap.parse_args()

    run_slip(Path(args.inp), Path(args.out))


if __name__ == "__main__":
    main()

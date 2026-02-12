"""
slip_from_csv.py

Reads a CSV of candidate bets and joins against either:
A) Slate preds (future games): data/processed/slate_<prop>_preds_<YYYY-MM-DD>.parquet
B) Historical preds (played games): data/processed/<prop>_with_pred_<season>_<seasontype>.parquet

Then computes:
- P(over line) assuming Normal(mu_pred, sigma)
- implied probability from odds
- edge, EV, fractional Kelly sizing
- BET / NO BET decision

Input CSV supports:

MODE (recommended):
  prop,season,season_type,player_name,game_date,line,odds_over
Optional:
  odds_under, kelly_multiplier, min_edge, event_id

If event_id is present, slate matching becomes extremely reliable.

Outputs: data/processed/betslip_results.csv by default
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from src.betting.edge_calculator import compute_edge, should_bet


# ----------------------------
# Paths
# ----------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _st(season_type: str) -> str:
    return season_type.replace(" ", "")


def _historical_pred_path(prop: str, season: str, season_type: str, root: Path) -> Path:
    st = _st(season_type)
    processed = root / "data" / "processed"
    return processed / f"{prop}_with_pred_{season}_{st}.parquet"


def _slate_pred_path(prop: str, date_str: str, root: Path) -> Path:
    processed = root / "data" / "processed"
    return processed / f"slate_{prop}_preds_{date_str}.parquet"


def _eval_path(prop: str, season: str, season_type: str, root: Path) -> Path:
    st = _st(season_type)
    return root / "models_artifacts" / f"{prop}_eval_{season}_{st}.json"


def _mu_col(prop: str) -> str:
    return {"points": "PTS_PRED", "rebounds": "REB_PRED", "assists": "AST_PRED"}[prop]


def _default_sigma(prop: str) -> float:
    return {"points": 4.0, "rebounds": 2.0, "assists": 1.7}[prop]


# ----------------------------
# Probability
# ----------------------------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _p_over_from_mu_sigma(mu: float, sigma: float, line: float) -> float:
    sigma = float(sigma)
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return float(1.0 - _normal_cdf(z))


def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


# ----------------------------
# Loading caches
# ----------------------------

def _load_sigma(prop: str, season: str, season_type: str, root: Path) -> float:
    p = _eval_path(prop, season, season_type, root)
    sigma = _default_sigma(prop)
    if p.exists():
        try:
            ev = json.loads(p.read_text())
            sigma = float(ev.get("rmse", sigma))
        except Exception:
            pass
    return sigma


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ----------------------------
# Matching
# ----------------------------

def _match_slate(df: pd.DataFrame, player_name: str, game_date: str, event_id: Optional[str]) -> pd.DataFrame:
    """
    Slate pred files contain future-game rows.

    Preferred match:
      EVENT_ID + PLAYER_NAME
    Fallback:
      GAME_DATE + PLAYER_NAME
    """
    df = df.copy()
    df["PLAYER_NAME_NORM"] = df["PLAYER_NAME"].astype(str).map(_norm_name)

    player_key = _norm_name(player_name)

    if event_id and "EVENT_ID" in df.columns:
        m = df[(df["PLAYER_NAME_NORM"] == player_key) & (df["EVENT_ID"].astype(str) == str(event_id))]
        if not m.empty:
            return m

    # fallback to date+name
    _ensure_datetime(df, "GAME_DATE")
    gd = pd.to_datetime(game_date).date()
    m = df[(df["PLAYER_NAME_NORM"] == player_key) & (df["GAME_DATE"].dt.date == gd)]
    return m


def _match_historical(df: pd.DataFrame, player_name: str, game_date: str) -> pd.DataFrame:
    df = df.copy()
    df["PLAYER_NAME_NORM"] = df["PLAYER_NAME"].astype(str).map(_norm_name)
    _ensure_datetime(df, "GAME_DATE")
    gd = pd.to_datetime(game_date).date()
    return df[(df["PLAYER_NAME_NORM"] == _norm_name(player_name)) & (df["GAME_DATE"].dt.date == gd)]


# ----------------------------
# Core runner
# ----------------------------

def run_slip(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    root = repo_root()
    bets = pd.read_csv(input_csv)

    # Defaults
    if "odds_under" not in bets.columns:
        bets["odds_under"] = np.nan
    if "kelly_multiplier" not in bets.columns:
        bets["kelly_multiplier"] = 0.25
    if "min_edge" not in bets.columns:
        bets["min_edge"] = 0.02
    if "event_id" not in bets.columns:
        bets["event_id"] = np.nan

    required = ["prop", "season", "season_type", "player_name", "game_date", "line", "odds_over"]
    missing = [c for c in required if c not in bets.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # caches
    slate_cache: Dict[Tuple[str, str], pd.DataFrame] = {}       # (prop, date) -> df
    hist_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}   # (prop, season, season_type) -> df
    sigma_cache: Dict[Tuple[str, str, str], float] = {}         # (prop, season, season_type) -> sigma

    rows = []

    for _, r in bets.iterrows():
        try:
            prop = str(r["prop"]).strip().lower()
            season = str(r["season"]).strip()
            season_type = str(r["season_type"]).strip()
            player_name = str(r["player_name"]).strip()
            game_date = str(r["game_date"]).strip()

            line = float(r["line"])
            odds_over = int(float(r["odds_over"]))
            odds_under = None if pd.isna(r["odds_under"]) else int(float(r["odds_under"]))

            k_mult = float(r["kelly_multiplier"])
            min_edge = float(r["min_edge"])
            event_id = None if pd.isna(r["event_id"]) else str(r["event_id"]).strip()

            # Determine whether to use slate preds:
            gd = pd.to_datetime(game_date).date()
            today = pd.Timestamp.today().date()
            use_slate = gd >= today  # future/today => slate; past => historical

            mu_col = _mu_col(prop)

            # sigma
            key_sig = (prop, season, season_type)
            if key_sig not in sigma_cache:
                sigma_cache[key_sig] = _load_sigma(prop, season, season_type, root)
            sigma = sigma_cache[key_sig]

            match = pd.DataFrame()

            if use_slate:
                key_s = (prop, game_date)
                if key_s not in slate_cache:
                    sp = _slate_pred_path(prop, game_date, root)
                    if not sp.exists():
                        raise FileNotFoundError(
                            f"Missing slate preds: {sp}\n"
                            f"Run: python -m src.slate.predict_slate --date {game_date} --prop {prop}"
                        )
                    slate_cache[key_s] = pd.read_parquet(sp)

                df = slate_cache[key_s]
                match = _match_slate(df, player_name=player_name, game_date=game_date, event_id=event_id)

            else:
                key_h = (prop, season, season_type)
                if key_h not in hist_cache:
                    hp = _historical_pred_path(prop, season, season_type, root)
                    if not hp.exists():
                        raise FileNotFoundError(
                            f"Missing historical preds: {hp}\n"
                            f"Run: python main.py predict-prop --prop {prop} --season {season}"
                        )
                    hist_cache[key_h] = pd.read_parquet(hp)

                df = hist_cache[key_h]
                match = _match_historical(df, player_name=player_name, game_date=game_date)

            if match.empty:
                rows.append(
                    {
                        "prop": prop,
                        "season": season,
                        "season_type": season_type,
                        "player_name": player_name,
                        "game_date": game_date,
                        "event_id": event_id or "",
                        "line": line,
                        "error": "No matching row in predictions parquet (slate/historical). Check player_name/date (and event_id if using slate).",
                    }
                )
                continue

            row0 = match.iloc[0]
            mu = float(row0[mu_col])

            # Use SIGMA_USED if present in slate file (we wrote it), else fallback to eval rmse
            sigma_used = float(row0.get("SIGMA_USED", sigma))

            p_over = _p_over_from_mu_sigma(mu=mu, sigma=sigma_used, line=line)

            over_res = compute_edge(
                p_over=p_over,
                odds_over=odds_over,
                odds_under=odds_under,
                side="over",
                kelly_multiplier=k_mult,
            )
            under_res = None
            if odds_under is not None:
                under_res = compute_edge(
                    p_over=p_over,
                    odds_over=odds_over,
                    odds_under=odds_under,
                    side="under",
                    kelly_multiplier=k_mult,
                )

            chosen = over_res
            if under_res is not None and under_res.ev_per_dollar > over_res.ev_per_dollar:
                chosen = under_res

            decision = "BET" if should_bet(chosen, min_edge=min_edge, min_ev=0.0) else "NO BET"

            rows.append(
                {
                    "prop": prop,
                    "season": season,
                    "season_type": season_type,
                    "player_name": str(row0.get("PLAYER_NAME", player_name)),
                    "game_date": str(pd.to_datetime(row0.get("GAME_DATE", game_date)).date()),
                    "player_id": row0.get("PLAYER_ID", ""),
                    "game_id": row0.get("GAME_ID", ""),           # historical
                    "event_id": row0.get("EVENT_ID", event_id or ""),  # slate
                    "line": line,
                    "sigma_used": sigma_used,
                    "mu_pred": mu,
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
                    "error": "",
                }
            )

        except Exception as e:
            rows.append(
                {
                    "prop": r.get("prop", ""),
                    "season": r.get("season", ""),
                    "season_type": r.get("season_type", ""),
                    "player_name": r.get("player_name", ""),
                    "game_date": r.get("game_date", ""),
                    "event_id": r.get("event_id", ""),
                    "line": r.get("line", ""),
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved betslip results -> {output_csv}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV of bets (bets_today.csv)")
    ap.add_argument("--out", dest="out", default="data/processed/betslip_results.csv", help="Output CSV path")
    args = ap.parse_args()
    run_slip(Path(args.inp), Path(args.out))


if __name__ == "__main__":
    main()

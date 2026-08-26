"""
export_static_site_data.py

Converts existing pipeline outputs (model eval JSONs, betslip_results.csv)
into a single static docs/data.json the GitHub Pages demo site reads.
Run this manually and commit the result whenever you want the public page
to reflect newer numbers -- there's no automation wired up for this yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"

PROPS = ["points", "rebounds", "assists"]

SLATE_COLUMNS = [
    "game_date",
    "player_name",
    "prop",
    "chosen_side",
    "line",
    "mu_pred",
    "edge",
    "ev_per_dollar",
    "decision",
]


def _load_model_performance(artifacts_dir: Path, season: str, season_type: str) -> list[dict]:
    st = season_type.replace(" ", "")
    rows = []
    for prop in PROPS:
        eval_path = artifacts_dir / f"{prop}_eval_{season}_{st}.json"
        if not eval_path.exists():
            continue

        ev = json.loads(eval_path.read_text())
        mae = ev["mae"]
        baseline_mae = ev.get("baseline_mae")
        lift_pct = round((baseline_mae - mae) / baseline_mae * 100, 2) if baseline_mae else None

        rows.append(
            {
                "prop": prop,
                "mae": round(mae, 3),
                "rmse": round(ev["rmse"], 3),
                "baseline_mae": round(baseline_mae, 3) if baseline_mae else None,
                "lift_pct": lift_pct,
                "rows_test": ev.get("rows_test"),
                "test_date_min": ev.get("test_date_min"),
                "test_date_max": ev.get("test_date_max"),
            }
        )
    return rows


def _load_todays_slate(processed_dir: Path) -> Optional[list[dict]]:
    path = processed_dir / "betslip_results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]
    if df.empty:
        return []

    keep = [c for c in SLATE_COLUMNS if c in df.columns]
    return df[keep].to_dict(orient="records")


def export_site_data(
    artifacts_dir: Path = ARTIFACTS_DIR,
    processed_dir: Path = DATA_PROCESSED,
    season: str = "2024-25",
    season_type: str = "Regular Season",
    out_path: Optional[Path] = None,
) -> dict:
    if out_path is None:
        out_path = DOCS_DIR / "data.json"

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "model_performance": _load_model_performance(artifacts_dir, season, season_type),
        "todays_slate": _load_todays_slate(processed_dir),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"Saved static site data -> {out_path}")
    return data


if __name__ == "__main__":
    export_site_data()

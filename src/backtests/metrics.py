"""
metrics.py: turn replayed lines into the numbers and verdict the pre-registered rules define
(design/backtest-rules.md, version 1).

Direction: the model says over if projection > line and under if projection < line. A line is a hit if the
actual result lands on that side. Accuracy: d = |line - actual| - |projection - actual|, positive when the
model's number was closer. Only scored, non-push lines with a direction are measured, and only one model
label at a time. Intervals are a bootstrap over whole games, since lines in a game move together.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.backtests.rules import (
    BOOT_SEED, BREAKEVEN, CI_PERCENTILES, GAP_BUCKETS, N_BOOT, VERDICT_MIN_LINES,
)


def _bootstrap(events: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Percentile interval of sum(num)/sum(den) when whole events are resampled."""
    _, inverse = np.unique(events, return_inverse=True)
    n_events = inverse.max() + 1
    per_num = np.bincount(inverse, weights=num, minlength=n_events)
    per_den = np.bincount(inverse, weights=den, minlength=n_events)
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.integers(0, n_events, size=(N_BOOT, n_events))
    ratios = per_num[draw].sum(axis=1) / per_den[draw].sum(axis=1)
    return np.percentile(ratios, CI_PERCENTILES)


def _interval(values) -> list:
    return [float(values[0]), float(values[1])]


def _bucket_rows(directional: pd.DataFrame) -> list:
    gap = (directional["projection"] - directional["line"]).abs()
    rows = []
    for lo, hi in GAP_BUCKETS:
        part = directional[(gap >= lo) & (gap < hi)]
        rows.append({
            "gap_from": lo, "gap_to": None if hi == float("inf") else hi, "n": int(len(part)),
            "hit_rate": float(part["hit"].mean()) if len(part) else None,
            "mean_d": float(part["d"].mean()) if len(part) else None,
        })
    return rows


def summarize(replayed: pd.DataFrame) -> Dict[str, object]:
    labels = replayed["model_label"].dropna().unique()
    if len(labels) > 1:
        raise ValueError(f"results cover more than one model ({sorted(labels)}); samples are never pooled")

    scored = replayed[replayed["status"] == "scored"].copy()
    line, proj, actual = (scored[c].astype(float) for c in ("line", "projection", "actual"))
    keep = (actual != line) & (proj != line)
    d = scored[keep].copy()
    d["over"] = d["projection"] > d["line"]
    d["hit"] = np.where(d["over"], d["actual"] > d["line"], d["actual"] < d["line"])
    d["d"] = (d["line"] - d["actual"]).abs() - (d["projection"] - d["actual"]).abs()

    n = int(len(d))
    out: Dict[str, object] = {
        "model_label": str(labels[0]) if len(labels) else None,
        "status_counts": {k: int(v) for k, v in replayed["status"].value_counts().items()},
        "n_scored": int(len(scored)),
        "n_directional": n,
        "lines_needed": max(0, VERDICT_MIN_LINES - n),
        "first_game": str(scored["game_date"].min()) if len(scored) else None,
        "last_game": str(scored["game_date"].max()) if len(scored) else None,
        "hit_rate": None, "hit_ci": None, "mean_d": None, "d_ci": None,
        "mae_model": None, "mae_market": None, "model_closer_share": None,
        "breakeven_mean": None, "mean_mins_before_tip": None, "buckets": [],
        "verdict": "interim",
    }
    if n == 0:
        return out

    events = d["event_id"].to_numpy()
    hits = d["hit"].to_numpy().astype(float)
    ds = d["d"].to_numpy()
    ones = np.ones(n)
    hit_ci = _interval(_bootstrap(events, hits, ones))
    d_ci = _interval(_bootstrap(events, ds, ones))
    out.update(
        hit_rate=float(hits.mean()), hit_ci=hit_ci, mean_d=float(ds.mean()), d_ci=d_ci,
        mae_model=float((d["projection"] - d["actual"]).abs().mean()),
        mae_market=float((d["line"] - d["actual"]).abs().mean()),
        model_closer_share=float((ds > 0).mean()),
        breakeven_mean=float(np.where(d["over"], d["be_over"], d["be_under"]).mean()),
        mean_mins_before_tip=float(d["mins_before_tip"].mean()),
        buckets=_bucket_rows(d),
    )
    if n >= VERDICT_MIN_LINES:
        out["verdict"] = "edge_shown" if (hit_ci[0] > BREAKEVEN and d_ci[0] > 0) else "no_edge_shown"
    return out

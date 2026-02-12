from __future__ import annotations

import pandas as pd


def confidence_tier(edge: float) -> str:
    if edge >= 0.10:
        return "A"
    if edge >= 0.05:
        return "B"
    if edge >= 0.02:
        return "C"
    return "D"


def format_betslip(csv_path: str, top_n: int = 25) -> None:
    df = pd.read_csv(csv_path)

    # keep only successful rows
    df = df[(df.get("error", "") == "")].copy()

    # only bets
    df = df[df["decision"] == "BET"].copy()
    if df.empty:
        print("No bets meet criteria today.")
        return

    # sort by edge descending
    df = df.sort_values("edge", ascending=False).head(top_n)

    date = df["game_date"].iloc[0]
    print(f"\n🔥 NBA MODEL SLATE — {date}\n")

    for i, r in enumerate(df.itertuples(index=False), 1):
        conf = confidence_tier(float(r.edge))
        odds = r.odds_over if r.chosen_side == "over" else r.odds_under

        print(f"{i}) {r.player_name} — {r.chosen_side.upper()} {r.line} {str(r.prop).title()} ({odds})")
        print(f"   Model: {round(float(r.mu_pred), 2)}")
        print(f"   Edge: +{round(float(r.edge) * 100, 1)}% | EV: +{round(float(r.ev_per_dollar), 2)} per $1")
        print(f"   Bet Size: {round(float(r.fractional_kelly) * 100, 1)}% bankroll | Confidence: {conf}\n")

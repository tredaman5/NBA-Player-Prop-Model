from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.ui.style import inject_global_styles, header_with_logo


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_CSV = REPO_ROOT / "data" / "processed" / "betslip_results.csv"


def confidence_tier(edge: float) -> str:
    if edge >= 0.10:
        return "A"
    if edge >= 0.05:
        return "B"
    if edge >= 0.02:
        return "C"
    return "D"


def tier_class(tier: str) -> str:
    return {"A": "tierA", "B": "tierB", "C": "tierC", "D": "tierD"}.get(tier, "tierD")


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV)

    # keep successful rows only
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""].copy()

    return df


def main():
    st.set_page_config(page_title="NBA Prop Edge Dashboard", page_icon="🏀", layout="wide")
    inject_global_styles()

    header_with_logo(
        "NBA Prop Edge Dashboard",
        "Daily slate → model probabilities → edge/EV → bet sizing (fractional Kelly).",
    )

    df = load_results()
    if df.empty:
        st.info("No results found yet. Run your daily pipeline first:\n\n`python -m src.ui.run_daily --date YYYY-MM-DD`")
        return

    # Sidebar controls
    st.sidebar.header("Filters")

    show_only_bets = st.sidebar.checkbox("Show only BET decisions", value=True)
    min_edge = st.sidebar.slider("Minimum edge (%)", 0.0, 20.0, 2.0, 0.5) / 100.0
    min_ev = st.sidebar.slider("Minimum EV per $1", -0.20, 0.50, 0.00, 0.01)
    prop_filter = st.sidebar.multiselect("Props", sorted(df["prop"].unique().tolist()), default=sorted(df["prop"].unique().tolist()))
    sort_by = st.sidebar.selectbox("Sort by", ["edge", "ev_per_dollar", "fractional_kelly"], index=0)
    top_n = st.sidebar.slider("Max rows", 10, 200, 60, 5)

    # Compute confidence tiers
    df["edge"] = df["edge"].astype(float)
    df["ev_per_dollar"] = df["ev_per_dollar"].astype(float)
    df["fractional_kelly"] = df["fractional_kelly"].astype(float)
    df["tier"] = df["edge"].map(confidence_tier)

    # Filters
    df2 = df[df["prop"].isin(prop_filter)].copy()
    df2 = df2[df2["edge"] >= min_edge]
    df2 = df2[df2["ev_per_dollar"] >= min_ev]
    if show_only_bets:
        df2 = df2[df2["decision"] == "BET"]

    # Sort
    df2 = df2.sort_values(sort_by, ascending=False).head(top_n)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df2):,}")
    col2.metric("A-tier", f"{(df2['tier'] == 'A').sum():,}")
    col3.metric("Avg Edge", f"{(df2['edge'].mean() * 100 if len(df2) else 0):.2f}%")
    col4.metric("Avg Bet Size", f"{(df2['fractional_kelly'].mean() * 100 if len(df2) else 0):.2f}%")

    st.divider()

    # Clean table view
    st.subheader("Bets (clean view)")

    clean = df2.copy()
    clean["Edge %"] = (clean["edge"] * 100).round(2)
    clean["Bet Size %"] = (clean["fractional_kelly"] * 100).round(2)
    clean["EV / $1"] = clean["ev_per_dollar"].round(3)

    keep_cols = [
        "game_date",
        "player_name",
        "prop",
        "chosen_side",
        "line",
        "mu_pred",
        "Edge %",
        "EV / $1",
        "Bet Size %",
        "tier",
        "odds_over",
        "odds_under",
        "decision",
    ]
    keep_cols = [c for c in keep_cols if c in clean.columns]
    st.dataframe(clean[keep_cols], use_container_width=True, height=420)

    st.divider()
    st.subheader("Card view (easy to read)")

    # Cards
    for r in df2.itertuples(index=False):
        tier = confidence_tier(float(r.edge))
        odds = r.odds_over if r.chosen_side == "over" else r.odds_under
        css = tier_class(tier)

        st.markdown(
            f"""
            <div class="card {css}">
              <div class="big">{r.player_name} — {str(r.prop).title()} ({r.chosen_side.upper()} {r.line})</div>
              <div class="muted">{r.game_date} • Odds: {odds} • Confidence: {tier}</div>
              <div style="margin-top:10px;">
                <span class="pill">Model: {float(r.mu_pred):.2f}</span>
                <span class="pill">Edge: +{float(r.edge)*100:.2f}%</span>
                <span class="pill">EV: +{float(r.ev_per_dollar):.3f}/$</span>
                <span class="pill">Bet Size: {float(r.fractional_kelly)*100:.2f}%</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Tip: keep your API key in an environment variable (ODDS_API_KEY). Don’t commit it.")


if __name__ == "__main__":
    main()

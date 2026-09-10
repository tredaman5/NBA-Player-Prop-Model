# 🏀 NBA Player Prop Edge Model

A fully automated end-to-end NBA player prop betting system that:

-   Pulls live sportsbook player props daily\
-   Generates pre-game predictions using machine learning\
-   Converts predictions into probabilistic edges\
-   Calculates expected value (EV) and optimal bet sizing\
-   Displays results in CLI format and a static web dashboard

**[Live demo](https://tredaman5.github.io/NBA-Player-Prop-Model/)** — a
public, read-only snapshot of current model performance and (once the
season starts) today's slate.

------------------------------------------------------------------------

## 🚀 What This Project Does

This project builds a complete quantitative workflow for NBA player
props:

1.  Fetch live odds (Points / Rebounds / Assists)\
2.  Build slate features for upcoming games\
3.  Predict player performance using trained ML models\
4.  Convert projections → probabilities (Normal distribution
    assumption)\
5.  Compare model probability vs sportsbook implied probability\
6.  Calculate edge, expected value, and fractional Kelly bet sizing\
7.  Display ranked bets in CLI and dashboard format

------------------------------------------------------------------------

## 🧠 Modeling Approach

### Feature Engineering

Models use:

-   Rolling per-minute production (3 / 5 / 10 game windows)
-   Predicted minutes model, including a rotation-teammates-out feature
    built from the official pregame inactive-player list
-   Opponent defense context (rolling points/rebounds/assists allowed)
-   Pace-normalized usage rate (share of team possessions used)
-   Team rolling statistics
-   Rest days & back-to-back indicators
-   Home / Away splits

Every rolling feature is shifted so a game's prediction only ever uses
games strictly before it — no feature is computed using information
that wouldn't have been available pregame.

### Prediction Flow

Historical Data → Feature Engineering → Model Training\
↓\
Slate Feature Generation\
↓\
Model Predictions\
↓\
Probability Conversion (Normal(mu, sigma))\
↓\
Edge + EV + Kelly Bet Sizing

------------------------------------------------------------------------

## 📊 Edge Calculation

Edge = Model Probability − Implied Probability

Expected value per \$1 wagered is computed using American odds.

Bet sizing uses:

-   Full Kelly Fraction
-   Fractional Kelly (default 0.25x)
-   Minimum edge threshold (default 2%)

------------------------------------------------------------------------

## 🗂 Project Structure

    NBA-Player-Prop-Model/
    │
    ├── app.py             # legacy Streamlit dashboard (being retired)
    ├── data/
    │   └── processed/
    ├── docs/              # static web dashboard (GitHub Pages)
    ├── models_artifacts/
    ├── tests/
    ├── src/
    │   ├── backtests/     # season-long prediction log + walk-forward backtest
    │   ├── betting/
    │   ├── common/        # shared feature-engineering: opponent, injury, usage-rate
    │   ├── odds/
    │   ├── slate/
    │   └── ui/

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/yourusername/NBA-Player-Prop-Model.git
    cd NBA-Player-Prop-Model

Install dependencies:

    pip install -r requirements.txt

Key libraries:

-   pandas
-   numpy
-   scikit-learn
-   nba_api
-   requests

------------------------------------------------------------------------

## 🔑 API Key Setup

This project uses The Odds API.

PowerShell:

    $env:ODDS_API_KEY="YOUR_KEY_HERE"

Do not commit your API key.

------------------------------------------------------------------------

## 🏃 Daily Run (One Command)

    python -m src.ui.run_daily --date 2026-02-13

This runs the full pipeline: - Fetch props - Build slate - Generate
predictions - Calculate edge & bet sizing - Output results

------------------------------------------------------------------------

## 📈 Dashboard

Results are published as a static web dashboard (`docs/`), hosted on
GitHub Pages: **<https://tredaman5.github.io/NBA-Player-Prop-Model/>**

It shows current model performance and, during the season, the day's
model-vs-line comparisons. To refresh it after a pipeline run:

    python -m src.ui.export_static_site_data   # regenerates docs/data.json
    git add docs/data.json && git commit -m "chore: refresh dashboard data"

A legacy Streamlit app (`app.py`) still exists for local interactive
filtering but is being retired in favour of the static dashboard.

------------------------------------------------------------------------

## 📌 Model Performance

  Prop       MAE    Baseline MAE   Lift
  ---------- ------ -------------- ------
  Points     4.77   4.82           +1.0%
  Rebounds   2.03   2.05           +0.8%
  Assists    1.40   1.40           -0.2%

Evaluated on the completed 2024-25 season with a chronological
(time-based) train/test split. "Baseline" is a naive last-5-games
rolling average; "Lift" is the model's improvement over that baseline.

**Methodology note:** an earlier version of the minutes model had a
data leakage bug — it used a game's own actual box score as a feature
to predict minutes *in that same game*, which isn't known until after
the game is played. That inflated its apparent accuracy substantially
(leaky MAE looked like 3.5; the honest number is 5.5, barely better
than the naive baseline). The bug is fixed, and the numbers above
reflect the honest, leakage-free pipeline. As the table shows, the
real lift over a naive baseline is currently modest, and assists
hasn't beaten baseline at all — this is measured against a naive
statistical baseline, **not** against real sportsbook lines yet. A
live backtest against actual market lines (`src/backtests/`) is built
and unit-tested, pending enough games from the next NBA season to
produce a meaningful sample.

------------------------------------------------------------------------

## 🧪 Testing

    pip install -r requirements.txt
    pytest tests/ -v

Tests cover leakage-safety of every rolling feature (each one is
verified to only use games strictly before the one it's predicting),
and the season-long prediction log's append/backfill logic.

------------------------------------------------------------------------

## 🔒 Disclaimer

This project is for educational and research purposes only.

It is not financial advice and does not guarantee profitability.

Sports betting involves financial risk.

------------------------------------------------------------------------

## 👤 Author

Tre Smith\
Computer Science --- University of South Carolina

Interests: - Sports analytics - Quantitative modeling - Machine learning
systems - Financial engineering

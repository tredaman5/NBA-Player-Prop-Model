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

**Known limitation:** the live slate builder reuses each player's
last-game feature row, so opponent context, predicted minutes, and
rolling windows are stale for tonight's game. Treat slate projections
as provisional until that's fixed.

------------------------------------------------------------------------

## 📥 Capturing Sportsbook Lines (free tier)

Sportsbook lines are the one input that can't be rebuilt later
(historical odds are paid), so they're captured separately from the
model, as an append-only log:

    python -m src.odds.capture_lines --dry-run     # free: shows what it would capture
    python -m src.odds.capture_lines               # captures games tipping off in the next 2 hours

The free Odds API tier is 500 credits/month. By default only
`player_points` is captured: 1 credit a game, roughly 8 a night and 230
a month, so every game all month fits (pass `--markets` for more, at 1
credit each per game). The tool only captures games close to tip-off
(which also gives lines nearest the close), skips games it already
captured, and stops at `--max-credits` (default 60 per run) or before
the remaining quota drops under `--reserve` (default 25). Listing games
is free, so checking often costs nothing.

Output goes to `data/raw/odds_lines_log.csv`, which is gitignored on
purpose: back it up somewhere private, and don't publish it — the
provider's terms prohibit redistributing their data as files or feeds.

### Automating it (GitHub Actions)

`.github/workflows/capture-lines.yml` runs the capture every 15 minutes
from late morning to late evening US Eastern, so it works without your
computer being on. This repo is public, so the log is saved to a
separate **private** repo. One-time setup:

1.  Create a private repo named `nba-prop-data` (tick "Add a README" so
    it isn't empty; the workflow expects a `main` branch).
2.  Create a fine-grained personal access token limited to that repo
    only, with **Contents: Read and write**. Note its expiry date: the
    workflow fails once it lapses.
3.  In this repo: Settings → Secrets and variables → Actions → add
    `ODDS_API_KEY` (your Odds API key) and `DATA_REPO_TOKEN` (the token).
4.  Actions tab → "Capture sportsbook lines" → Run workflow. Manual runs
    are a dry run by default: it lists tonight's games and spends no
    credits. Once that looks right, scheduled runs take over.

GitHub pauses scheduled workflows in a repo with no activity for 60
days, so push something now and then, or re-enable it under the Actions
tab.

------------------------------------------------------------------------

## 📈 Dashboard

Results are published as a static web dashboard (`docs/`), hosted on
GitHub Pages: **<https://tredaman5.github.io/NBA-Player-Prop-Model/>**

It shows current model performance and, during the season, the day's
model-vs-line comparisons. To refresh it after a pipeline run:

    python -m src.ui.export_static_site_data   # regenerates docs/data.json
    git add docs/data.json && git commit -m "chore: refresh dashboard data"

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

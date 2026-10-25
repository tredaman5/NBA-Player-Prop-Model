# 🏀 NBA Player Prop Edge Model

An NBA player prop projection pipeline that:

-   Captures sportsbook player-points lines automatically (free tier)
-   Projects each player's points, rebounds and assists with machine learning
-   Prints tonight's projection next to the median posted line, with the difference
-   Checks its own inputs and stops loudly when they look wrong

It does not tell you what to bet. The backtest against real lines is
still accumulating, and until it shows the model beats the market the
output stays a plain projection-versus-line comparison.

**[Live demo](https://tredaman5.github.io/NBA-Player-Prop-Model/)** — a
public, read-only snapshot of current model performance and (once the
season starts) today's slate.

------------------------------------------------------------------------

## 🚀 What This Project Does

1.  Capture sportsbook lines for every game (GitHub Actions, private data repo)
2.  Build features for tonight's games with the same code training uses
3.  Predict minutes, then points / rebounds / assists
4.  Join the median line across books and show the difference
5.  Run checks and write a CSV plus a run summary

------------------------------------------------------------------------

## 🧠 Modeling Approach

### Feature Engineering

Models use:

-   Rolling per-minute production (3 / 5 / 10 game windows)
-   A predicted-minutes model (rest, home/away, rolling minutes) that
    feeds every prop model
-   Opponent defense context (rolling points/rebounds/assists allowed)
-   Pace-normalized usage rate (share of team possessions used)
-   Team rolling points (points model only)
-   Rest days & back-to-back indicators
-   Home / Away splits

Every rolling feature is shifted so a game's prediction only ever uses
games strictly before it — no feature is computed using information
that wouldn't have been available pregame.

### Prediction Flow

Historical games → one feature builder → trained models\
↓\
Tonight's games (from the captured lines and current rosters) → the same feature builder\
↓\
Projection per player, next to the median line and the difference

------------------------------------------------------------------------

## 📊 Edge, EV and Kelly (not used yet)

`src/betting/edge_calculator.py` has tested edge, expected-value and
fractional-Kelly math. Nothing runs it or shows its output: turning a
projection into a probability and a bet size means claiming the model
beats the market, and the backtest hasn't shown that yet.

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
    │   ├── betting/       # edge/EV/Kelly math (tested, not run or shown yet)
    │   ├── common/        # shared feature engineering (opponent, usage rate) + data-quality gates
    │   ├── odds/
    │   ├── pipeline/      # the daily command and its checks
    │   ├── serving/       # live slate: lines + rosters + models -> projections
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

This project uses The Odds API (free key: <https://the-odds-api.com>).
Put your own key in a local `.env` file once and every command picks it
up, with no exporting in each shell:

    cp .env.example .env      # then edit .env and replace your_key_here

`.env` is gitignored; never commit it, and don't share your key (each
person uses their own). A real `export ODDS_API_KEY=...` in your shell
still overrides the file.

`.env` is for running on your own machine only. The scheduled GitHub
Actions capture (below) runs on GitHub's servers and can't see your
`.env`, so it reads the key from GitHub's encrypted repository secrets
instead.

------------------------------------------------------------------------

## 🏃 Daily Run (One Command)

    python -m src.pipeline.daily                                       # tonight (US Eastern)
    python -m src.pipeline.daily --date 2026-10-23 --include-started   # replay a past night

Set `LINES_LOG_PATH` in `.env` to your clone of the private data repo
(or pass `--log-path`). The run refreshes the current season's game
history, loads current rosters and the newest models, builds the slate,
checks it, and writes two files to `data/processed/` (gitignored):
`slate_<date>.csv`, `slate_<date>.html` (a searchable, sortable page)
and `run_<date>.json` (counts, skip reasons, warnings; written on
failure too).

For every player with a posted points line from at least 3 books, the
slate shows the model's projection, the median line across books, and
the difference (projection minus line). History is cut off at the slate
date, so replaying a past night can't see that night's results. Every
player left off (line from fewer than 3 books, fewer than 5 prior games,
name not on either team's roster) is listed with the reason.

It stops instead of producing a bad slate when:

-   the lines log has no games on that date (exit code 2),
-   more than 20% of the day's lines can't be matched to a team or
    player, which usually means a provider renamed something (exit 1),
-   the game history is more than 3 days old inside a season, or the
    roster is missing a team (exit 1).

Old models only produce a warning (over 60 days since training data
ended): retraining cadence is a separate decision.

There is no injury feature yet, so a big gap between projection and line
often just means the market knows about a lineup change the model
doesn't. The output shows no edge, probability or bet.

------------------------------------------------------------------------

## 🤖 Running the Slate Automatically

**On your Mac, not GitHub Actions.** A daily workflow was built and tried
(2026-10-24): it reached the pipeline, then failed because
stats.nba.com does not answer GitHub's servers (`Read timed out` on the
game-log call). The line capture works on Actions because it only talks
to The Odds API; the slate needs NBA game history, so it runs on your
own machine.

    bash scripts/daily_local.sh        # run it by hand
    bash scripts/install_schedule.sh   # run it daily at 4:30 pm and 6:30 pm (local time)

`daily_local.sh` runs `git pull` in your data-repo clone (default
`~/Projects/nba-prop-data`, or set `NBA_DATA_REPO`), builds the slate with
the models in that clone's `models/`, and saves the page and a run
summary to its `slates/` folder: open `slates/latest.html`. It never
commits or pushes. The page holds real sportsbook lines, so it stays on
your machine. The output is logged to `~/Library/Logs/nba-slate.log`;
a failed run also raises a macOS notification, and a quiet day (no games
or no captured lines) just logs and exits.

Things to know about the schedule:

-   It only runs while the Mac is awake. A run missed during sleep happens
    when the Mac wakes; a slate for a day the Mac was off all afternoon
    is lost.
-   It skips (and logs) when unplugged below 20% battery
    (`NBA_MIN_BATTERY` changes the limit).
-   Remove it with `bash scripts/install_schedule.sh --uninstall`.
-   Retrain locally and copy the new models into the data repo's
    `models/` folder when you want them updated; nothing retrains
    automatically.

------------------------------------------------------------------------

## 🧪 Training

    python main.py train-all --seasons 2024-25 2025-26

Builds features once over a continuous multi-season history (rolling
windows and rest days carry over the offseason), trains the minutes
model and the points/rebounds/assists models, and holds out the most
recent season's last 20% by date. Features come from
`src/features/builder.py`, the single feature builder; a parity test
checks that features built for an upcoming game equal the ones training
computes for the same game.

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
credit each per game). The tool only captures games close to tip-off,
skips games it already captured, and stops at `--max-credits` (default 60 per run) or before
the remaining quota drops under `--reserve` (default 25). Listing games
is free, so checking often costs nothing.

Output goes to `data/raw/odds_lines_log.csv`, which is gitignored on
purpose: back it up somewhere private, and don't publish it — the
provider's terms prohibit redistributing their data as files or feeds.

### Automating it (GitHub Actions)

`.github/workflows/capture-lines.yml` runs the capture on a schedule
(every 15 minutes, late morning to late evening US Eastern), so it
works without your computer being on. GitHub treats scheduled runs as
best-effort and in practice they fire every few hours, so each run
captures any game tipping off within the next 4 hours, and the log
records both capture time and tip-off time. This repo is public, so the log is saved to a
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

| Prop     | MAE  | Baseline MAE | Lift   |
|----------|------|--------------|--------|
| Points   | 4.79 | 4.84         | +1.10% |
| Rebounds | 1.92 | 1.94         | +0.86% |
| Assists  | 1.41 | 1.43         | +0.85% |

Trained on 2024-25 plus the first 80% of 2025-26 and evaluated on the
last 20% of 2025-26 (a chronological split by date, never random).
"Baseline" is a naive last-5-games rolling average; "Lift" is the
model's improvement over that baseline. Training on two seasons
instead of one helped rebounds, assists and minutes and left points
about the same (single-season training gave points +1.2%, rebounds
+0.4%, assists +0.6%). The lift is small, so treat it as roughly "no
better than a rolling average" until proven otherwise. This is
measured against a naive statistical baseline, **not** against real
sportsbook lines yet. A live backtest against actual market lines
(`src/backtests/`) is built and unit-tested, pending enough captured
games to produce a meaningful sample.

**Methodology notes**

-   **Leak #1 (fixed).** An earlier minutes model used a game's own
    box score as features to predict minutes *in that same game*. That
    made its error look like 3.5 minutes; the honest number is about
    5.5, barely better than the baseline.
-   **Leak #2 (caught and rejected).** Knowing which rotation players
    are out tonight genuinely helps (+1.4 points of lift in 2024-25),
    but only via the official pregame inactive list, which the data
    provider stopped serving for 2025-26. The obvious substitute,
    "rotation players with no minutes in tonight's box score", scored
    +7.2% against the official list's +3.8% on identical data, because
    "who didn't play" is the outcome. "Who was absent last game" is
    leak-free but added only noise. So the models use no injury
    feature until a real pregame source exists.
-   **Stage-1 inputs (fixed; effect negligible).** The prop models use a
    predicted-minutes feature. For training rows it used to come from a
    model that had trained on those same rows, so it was more accurate
    there (4.97 minutes of error) than on new games (5.32), and the prop
    models could over-trust it. It now comes from models trained only on
    earlier games (6 time blocks, settings fixed before looking at
    results), which cuts that gap to 5.08 vs 5.32. Accuracy barely
    moved: points +1.03% to +1.10%, rebounds +0.96% to +0.86%, assists
    +0.80% to +0.85%, all within noise. It's a correctness fix, not an
    improvement, and it costs about 16% of the training rows.
-   **Data-quality gates.** That 2025-26 failure also exposed a
    pipeline flaw: a third of the calls failed, the rest returned
    nothing, and the build trained on it anyway. Every data pull is
    now validated (schema, duplicates, impossible values, game
    pairing) before it's cached, and training refuses to run on a
    feature that is missing, constant, or mostly empty. On its first
    run the gate also found that the rebounds and assists models had
    always been listing two team-points features they never received.

------------------------------------------------------------------------

## 🧪 Testing

    pip install -r requirements.txt
    pytest tests/ -v

Tests cover leakage-safety of every rolling feature (each one is
verified to only use games strictly before the one it's predicting),
the data-quality gates, the line-capture budget guards, and the
season-long prediction log's append/backfill logic.

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

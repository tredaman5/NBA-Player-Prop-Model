# Slate rewrite: season-ready, honest daily projections

Status: proposed, 2026-10-21. Replaces `run_daily.py` and the slate builder.

## Why

`run_daily.py` was written for one season and one provider. Measured problems:

| Problem | Evidence |
|---|---|
| Live features are stale. `build_slate.py` copies each player's last-game row and overwrites only event, home/away and rest. Opponent context, `MIN_PRED` and every rolling window are a game behind. | Code: `build_slate.py:200-206` |
| Teams come from each player's last game. | 110 of 436 rostered players (25%) are on a different team than their last 2025-26 game, including 63 of 266 rotation regulars. 83 rostered players have no 2025-26 log at all. |
| Season is hardcoded (`2025-26`) and every season is built in isolation, so rolling windows restart each October. | Defaults in `build_slate`, `predict_slate`, `fetch_bets_theoddsapi` |
| The prop models train on an optimistic `MIN_PRED`: stage-1 predictions for the rows they train on are in-sample. | Minutes error 4.70 on rows the minutes model trained on vs 5.31 on unseen rows |
| Odds are fetched a second time at 3 credits a game, against 1 for the capture. The free tier can't afford both. | `fetch_bets_theoddsapi` defaults to 3 markets |
| Stages talk through filenames and `subprocess`. | `run_daily.py` |

Part of this was self-inflicted: this session added features to training without touching serving. The design therefore makes **one feature code path** for training and serving, and proves they agree with a test.

## Goals and non-goals

Goals
1. One feature builder used by training, serving and the backtest, with a parity test.
2. Continuous history across seasons (no October reset).
3. Correct team and opponent for tonight, from current rosters.
4. Honest stage-1 inputs (out-of-fold `MIN_PRED`).
5. No second odds fetch: read the lines the capture already saved.
6. Automatic daily publish to the Pages site: projection, line, difference. Nothing else.

Non-goals (separate work): an injury/availability feature (needs a real pregame source), calibrated probabilities, accounts or a backend, "edge"/"BET" language, more props than points on the free credit tier.

## Approaches considered

- **A. Patch `build_slate`** to recompute the stale columns. Smallest change, but it keeps two code paths, so skew returns the next time a feature is added. It just happened.
- **B. One builder over history plus placeholder rows for tonight's games (recommended).** Rolling features already use `shift(1)`, so an unplayed game sees only the past. Same code, no skew by construction.
- **C. A feature store/database.** Solves a scale problem we don't have.

## Design (approach B)

**History.** One continuous player-game log and team-game log across the last 3 completed seasons plus the current one (we hold 2 completed so far), refreshed daily from the league-wide logs (one call each). Every pull goes through `validation.py` before it is used.

**Feature builder** (`src/features/`, pure functions over DataFrames, no file I/O):
`build_features(player_games, team_games, upcoming=None)`. For `upcoming` games it appends rows with unknown box-score stats and the right team, opponent and home/away, then runs the same steps training runs: context, rest days, rolling minutes, per-minute rates, usage, team points, opponent defense. Rows with missing targets are kept through feature building and dropped only when training.

**Parity test (the key test).** For sampled games, including one early in a season and one right after the offseason gap, features built "as upcoming" from history truncated before the game must equal the training features for that same game.

**Stage 1.** `MIN_PRED` for training rows comes from time-blocked, expanding-window folds, so no row is predicted by a model that trained on it. Serving uses the model trained on all history.

**Training.** All seasons in the history window, chronological holdout on the most recent block. Measured on identical 2025-26 test games: points lift +1.15% (one season) vs +1.49% (two); minutes +1.69% vs +2.11%. Each model saves metadata (feature list, seasons, date, metrics, git SHA).

**Tonight's games and players.**
- Games: from the captured log (event id, teams, tip-off). The 30-team name map already exists.
- Players: the players with a posted line. A posted line is also a rough availability filter, since books usually pull props for players ruled out (late scratches still slip through).
- Team: current rosters (`CommonAllPlayers`, 519 players, all 30 teams verified), not last game.
- Names: normalized match to the roster. Unmatched names are listed in the run output, never silently dropped.
- A projection requires at least 5 prior games across seasons. Rookies are skipped with the reason shown.

**Lines.** Read the latest capture per (game, player, market) from the private log. Publish the median line only: no per-book prices, no odds history.

**Freshness gate.** Refuse to publish if the newest game in history is more than 3 days old.

**Orchestration.** `python -m src.pipeline.daily` calls functions in order (refresh, validate, features, predict, join lines, render). No `subprocess`. `run_daily.py` and `slip_from_csv.py` retire; `edge_calculator.py` stays as tested math but is not run or shown.

**Publishing.** A second workflow triggered by the capture workflow finishing (`workflow_run`), not by its own cron, since GitHub's cron fires hours apart. It reads models and history from the private data repo, builds the page, and deploys. The slate is rendered into the page rather than shipped as a standalone data file of lines.

**Backtest.** Because of the parity test, the walk-forward backtest can regenerate "what we would have projected" for every line in the log, as of that game's date. It no longer depends on a live prediction having been logged.

## Milestones (each shippable, each ends with tests and an honest metric)

1. Unified feature builder, parity test, cross-season history, retrain on both seasons.
2. Out-of-fold `MIN_PRED`; measure the change honestly (it may be small).
3. Upcoming-game builder from the captured log and rosters; produce a slate locally.
4. `src/pipeline/daily.py` with the freshness gate; retire `run_daily.py`.
5. Publish workflow and slate page (search, sort, filter).
6. Backtest integration.

## Decisions for the owner

- **Pages deploy method** (needed at milestone 5): GitHub's Pages deploy action keeps `main` free of daily bot commits but needs one Pages setting changed; committing a rendered file to `main` needs no setting but adds about 180 commits a season. Recommended: the deploy action.
- **Minimum history** for a projection: 5 games (proposed).
- **Page content**: projection, median line, difference, and the backtest status. No edge or BET language until the backtest supports it.

## Risks

- nba_api endpoints change or degrade (the inactive-list failure). Mitigation: validation gates and a freshness gate.
- Name mismatches between the odds provider and nba_api. Mitigation: unmatched names are reported.
- Stars out tonight make teammates' projections worse and we have no pregame availability data. Mitigation: a visible limitation; revisit with the NBA's official injury report.
- GitHub scheduled runs are unreliable. Mitigation: event-driven publish.
- Free credit tier covers points only.

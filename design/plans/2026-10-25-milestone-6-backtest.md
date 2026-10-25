# Milestone 6: Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every captured line against what actually happened, using what the model would have projected from information before that game, and report the result under the rules in `design/backtest-rules.md` (committed first, as `f8dd34c`).

**Architecture:** Because training and serving share one feature builder, the as-of features for a logged game are exactly the training-frame row for that player-game: build features once over the full history and look them up. `src/backtests/replay.py` joins each consensus line to its box-score row and projection and records a status for every line; `src/backtests/metrics.py` turns the scored lines into the numbers and verdict the rules define; `src/backtests/report.py` is the command; the daily pipeline calls the same functions and shows a status line on the page.

**Tech Stack:** pandas, numpy, existing models and builder. No new dependencies.

**Spec:** `design/2026-10-21-slate-rewrite.md` (milestone 6 of 6) and `design/backtest-rules.md`.

Commands assume the project virtualenv (`.venv`) and run from the repo root. Every code task is test-first.

## Global Constraints

- The numbers in `src/backtests/rules.py` must equal the ones in `design/backtest-rules.md`; a test enforces it. Changing a rule means a new dated version of that file first.
- A projection for a game uses only games strictly before it; a test proves the game's own box score cannot change it.
- A line in a game inside the model's training window is never scored.
- No "edge", "bet", EV or "pick" language anywhere unless the verdict is `edge_shown`.
- Line-level results hold sportsbook lines: they go to `data/processed/` (gitignored), never `docs/`.
- Specs and plans live in `design/`.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

- Modify `src/serving/lines.py` (`consensus_lines` adds `be_over`, `be_under`).
- Create `src/backtests/rules.py`, `src/backtests/replay.py`, `src/backtests/metrics.py`, `src/backtests/report.py`; tests `tests/test_backtest_rules.py`, `tests/test_backtest_replay.py`, `tests/test_backtest_metrics.py`.
- Modify `src/pipeline/daily.py`, `src/pipeline/page.py` (backtest status), their tests, `README.md`.
- Delete `src/backtests/snapshot_logger.py`, `src/backtests/walk_forward.py`, `tests/test_snapshot_logger.py` (the old betslip-log backtest the new one replaces; nothing imports them).

---

### Task 1: Rules as code

**Interfaces**
- Produces `src/backtests/rules.py`: `FIRST_GAME_DATE = dt.date(2026, 10, 26)`, `MIN_PRIOR_GAMES = 5`, `VERDICT_MIN_LINES = 1500`, `BREAKEVEN = 0.524`, `N_BOOT = 5000`, `BOOT_SEED = 2026`, `CI_PERCENTILES = (2.5, 97.5)`, `GAP_BUCKETS = ((0, 1), (1, 2), (2, 3), (3, float("inf")))`.

- [ ] Test (`tests/test_backtest_rules.py`): read `design/backtest-rules.md` and assert it contains `2026-10-26`, `1,500`, `0.524`, `5,000`, `seed 2026`, "at least 5 prior games", and that each constant above equals the value stated there.
- [ ] Run, expect FAIL (no module). Implement `rules.py`. Run, expect PASS. Commit `feat(backtest): the pre-registered rules as constants`.

### Task 2: Replay every logged line

**Interfaces**
- Consumes: `consensus_lines`, `normalize_name`, `match_to_roster`, `TEAM_ABBR`, `MARKET_TO_PROP`, `Models`, `build_features`, `validate_serving_frame`, `PROP_TARGETS`.
- Produces `replay_lines(log, players, teams, models, min_books=MIN_BOOKS, min_prior=MIN_PRIOR_GAMES, first_game_date=FIRST_GAME_DATE) -> DataFrame`, one row per consensus line (markets in `MARKET_TO_PROP` only), columns `event_id, game_date (date), player, player_id, team, market, line, books, mins_before_tip, be_over, be_under, projection, min_pred, prior_games, actual, status, model_label`. `status` is exactly one of `scored`, `unknown team name`, `before sample start`, `in training window`, `thin market`, `game not played yet`, `did not play`, `name not found`, `few prior games`, applied in that order. Only `scored` rows have `projection`, `actual`.
- Details: the game date is the commence time in US Eastern. "Game not played yet" means neither team has a box-score row on that date. A name matches within the event's two teams against that date's rows with minutes above 0; if it does not match but the normalized name and team exist anywhere in history it is `did not play`, otherwise `name not found`. `prior_games` is the number of earlier games with minutes above 0. Projections: minutes model gives `MIN_PRED` (clipped 0 to 60) and the prop model gives the projection (clipped 0 to 80), after `validate_serving_frame`.

- [ ] Tests (`tests/test_backtest_replay.py`) on the synthetic league extended to 13 games (game 13 is on 2025-01-25, BOS home), with a log row set for that game and `first_game_date` set so it is in the sample, and models with `trained_through` before it:
  - Every status above is produced by a crafted line (unknown home team name, date before the sample start, date on or before `trained_through`, 2 books, an event on a date with no box scores, a known player who has no row that night, a made-up name, a player with fewer than `min_prior` games).
  - **Parity:** the projection for a scored line equals `build_slate(...)` run for 2025-01-25 on the history before that date, for the same player.
  - **No peeking:** adding 50 points to a scored player's actual box score in history changes `actual` by 50 and leaves `projection` unchanged.
  - `actual` equals that player's PTS in that game; row count equals the number of consensus lines in the log.
- [ ] Run, expect FAIL. Implement. Run, expect PASS. Commit `feat(backtest): replay every logged line against the box scores`.

### Task 3: Metrics and the verdict

**Interfaces**
- Produces `summarize(replayed: DataFrame) -> dict` with keys `model_label`, `status_counts`, `n_scored`, `n_directional` (scored, actual != line, projection != line), `hit_rate`, `hit_ci`, `mean_d`, `d_ci`, `mae_model`, `mae_market`, `model_closer_share`, `breakeven_mean`, `mean_mins_before_tip`, `first_game`, `last_game`, `buckets` (exploratory list), `verdict` in `interim | edge_shown | no_edge_shown`, `lines_needed` (`VERDICT_MIN_LINES - n_directional`, floored at 0). Verdict is `interim` while `n_directional < VERDICT_MIN_LINES`; otherwise `edge_shown` only if `hit_ci[0] > BREAKEVEN` and `d_ci[0] > 0`.
- The bootstrap resamples events (`event_id`) with `N_BOOT` draws, `BOOT_SEED`, percentiles from `CI_PERCENTILES`. `hit`: over if projection > line and actual > line, or under if projection < line and actual < line. `d = |line - actual| - |projection - actual|`.

- [ ] Tests (`tests/test_backtest_metrics.py`) on hand-built frames: hit rate and `d` computed exactly on a small frame; pushes and `projection == line` excluded; an empty or all-unscored frame returns zeros and `interim`; with 1,499 directional lines the verdict is `interim` even when perfect; with 1,500 perfect-direction, closer-than-line lines it is `edge_shown`; with 1,500 coin-flip lines it is `no_edge_shown`; only one of the two conditions holding gives `no_edge_shown`; the bootstrap is reproducible (two calls equal) and the interval contains the point estimate; buckets partition the directional lines by `abs(projection - line)`.
- [ ] Run, expect FAIL. Implement. Run, expect PASS. Commit `feat(backtest): hit rate, accuracy and the pre-registered verdict`.

### Task 4: The command, the pipeline status line, cleanup

**Interfaces**
- `src/backtests/report.py`: `evaluate(log, players, teams, models) -> (replayed, summary)`; `render_report(summary) -> str`; `main(argv)` with `--log-path`, `--models-dir`, `--out-dir`; writes `backtest_<label>.csv` (line level) and `backtest_<label>.json` to the out dir.
- `run_pipeline` calls `evaluate` with the history it already loaded, adds `summary["backtest"]` (the summary dict without line-level data); an exception in the backtest is recorded as a warning (`backtest failed: ...`) and never fails the slate.
- The page shows one status line: for `interim`, "Backtest (interim, not a verdict): X of 1,500 lines scored so far for model <label>"; its numbers show with their intervals; for `no_edge_shown` it says the model did not beat the lines; for `edge_shown` it states the result with intervals. No betting language unless `edge_shown`.

- [ ] Tests: `render_report` for interim output contains "interim", the label, the line count and none of edge/bet/EV/pick; `run_pipeline` result summary includes a `backtest` dict; a forced backtest exception becomes a warning and the slate still writes; the page contains the status line.
- [ ] Implement; delete the three old backtest files; add a README "Backtest" section (the rules file, the command, how to read interim vs verdict, the 1,500 threshold, that this only reflects games after 2026-10-26). Run the full suite. Commit `feat(backtest): report command and a status line on the slate page`.

### Task 5: Real-data run and finish

- [ ] Run `python -m src.backtests.report` on the private log and the local models. Check: every line has a status and counts add up to the consensus lines; on dates before the sample start everything is `before sample start`; a handful of `scored` rows (once games after 2026-10-26 exist) have `actual` equal to the real box-score points; the `name not found` rate is small and the unmatched names are listed so a mapping problem is visible.
- [ ] To exercise the full path before the sample window opens, run once with `--first-game-date 2026-10-20` against the log and report the output clearly marked as a dry run (an exploratory look at pre-registration-excluded games; never reported as a result).
- [ ] Full suite, `git status` clean, then use superpowers:finishing-a-development-branch. Push only on "push".

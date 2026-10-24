# Milestone 3: Live Slate Implementation Plan

**Goal:** One command prints tonight's slate: for each player with a posted line, the model's projection, the median sportsbook line, and the difference, built from the captured lines and current rosters through the same `build_features` that training uses.

**Spec:** `design/2026-10-21-slate-rewrite.md` (milestone 3 of 6). Replaces the old `src/slate/` path (retired in milestone 4).

**Tech stack:** pandas, numpy, scikit-learn, nba_api, joblib, pytest. No new dependencies.

Commands assume the project virtualenv is active and are run from the repo root. Every task is test-first: write the failing test, watch it fail, implement, watch it pass, commit.

## Global Constraints

- Features for tonight come only from `src/features/builder.py::build_features` (no feature logic elsewhere).
- History passed to the builder is truncated to games strictly before the slate date, so a replay of a past date can never see that night's results.
- Every network pull is validated before use; the slate never silently drops a player: skipped players are listed with a reason.
- No odds data leaves the machine: output goes to `data/processed/` (gitignored), never to `docs/`.
- No "edge", "BET", EV or probability anywhere in the output. Projection, line, difference only.
- Specs and plans live in `design/`, not `docs/`.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Decisions (made from the first-look analysis, 2026-10-24)

| Decision | Why |
|---|---|
| Per book, use the line priced closest to even odds; then the median across books | Bovada posts several alternate lines per player; 448 of 2,669 book-player pairs had more than one |
| A line needs at least 3 books | Thin single-book markets (mostly bench players) are noise |
| Match names to the roster restricted to the event's two teams | Handles trades, duplicate names across teams, and gives a precise "not on either roster" reason |
| Player needs at least 5 prior games | Rookies and new players have no usable history; 14 of 406 line-players were skipped in the first look |
| Refresh the current season's history on every run (never read its cache) | A cache written mid-season goes stale; found in the first look |
| Tolerate an empty current season (opening day) | Otherwise the first run of every October crashes |
| Freshness gate: if the newest game before the slate date is in the same season and more than 3 days old, refuse | A stale history means wrong rolling features |
| Report model age (`trained_through`) but do not gate on it | Retraining cadence is a later decision |

---

### Task 1: Team names and player-name matching

**Files:** create `src/common/teams.py`, `src/serving/__init__.py`, `src/serving/names.py`; test `tests/test_serving_names.py`.

**Interfaces**
- `TEAM_ABBR: dict[str, str]` maps Odds API team names (including both Clippers spellings) to NBA abbreviations (30 teams).
- `normalize_name(name: str) -> str`: lowercase, strip accents, drop periods/apostrophes, hyphen to space, drop suffix tokens (jr, sr, ii, iii, iv, v).
- `match_to_roster(lines: DataFrame, roster: DataFrame) -> DataFrame`: `lines` needs `player, home_abbr, away_abbr`; `roster` needs `PERSON_ID, DISPLAY_FIRST_LAST, TEAM_ID, TEAM_ABBREVIATION`. Returns `lines` plus `PLAYER_ID, TEAM_ID, TEAM_ABBREVIATION, match_status` where `match_status` is `matched`, `not_on_either_roster`, or `ambiguous`.

**Tests:** accents/suffix/period normalization (`Nikola Jokić` == `Nikola Jokic`; `Jaren Jackson Jr.` == `Jaren Jackson`); a player on a team that is not in the event is `not_on_either_roster`; two roster players with the same name on different teams resolve to the one on the event's teams; two on the same event's teams is `ambiguous`; all 30 team names present.

### Task 2: Reading the captured lines

**Files:** create `src/serving/lines.py`; test `tests/test_serving_lines.py`.

**Interfaces**
- `load_line_log(path) -> DataFrame`: parses dates (UTC-aware), requires the 11 log columns, rejects an empty file.
- `main_lines(log) -> DataFrame`: one row per `(event_id, bookmaker, market, player)`, choosing the line whose over/under implied probabilities are closest; rows missing a price are dropped first.
- `consensus_lines(log, min_books=3) -> DataFrame`: uses only each event's latest `captured_at`; columns `event_id, market, player, line (median across books), books, captured_at, commence_time, home_team, away_team, thin_market (bool)`.

**Tests:** alternate lines: the balanced one wins; median across books; a player with only 2 books is flagged `thin_market`; only the latest capture of an event is used; missing prices dropped; empty log rejected.

### Task 3: History refresh and empty-season tolerance

**Files:** modify `src/common/validation.py` (add `EmptyDataError(DataQualityError)`, raised for empty frames), `src/features/history.py` (`refresh_seasons` parameter); tests in `tests/test_validation.py`, `tests/test_features_history.py`.

**Interfaces**
- `load_history(seasons, season_type, use_cache, fetch_players, fetch_teams, refresh_seasons=())`: seasons in `refresh_seasons` are fetched with `use_cache=False`; if a refresh season is empty (`EmptyDataError`) it is skipped, not fatal. An empty season that is not a refresh season still raises.

**Tests:** refresh seasons are fetched with `use_cache=False`, others with the caller's value; an empty refresh season is skipped; an empty non-refresh season raises; existing tests still pass.

### Task 4: Model loading and a serving-time data check

**Files:** create `src/serving/models.py`; modify `src/common/validation.py` (add `validate_serving_frame`); tests `tests/test_serving_models.py`, `tests/test_validation.py`.

**Interfaces**
- `load_models(artifacts_dir=ARTIFACTS_DIR, label=None) -> Models`, a dataclass with `minutes, props (dict prop -> model), features (dict name -> list), rmse (dict prop -> float), trained_through: str, label: str`. With `label=None` it picks the newest label that has all four models; a missing model raises a clear error naming the file.
- `MARKET_TO_PROP = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists"}`.
- `validate_serving_frame(df, features, max_null_share=0.5)`: raises `DataQualityError` if a feature column is missing or more than `max_null_share` of its values are null (constant columns are allowed: a one-game slate legitimately has them).

**Tests:** newest label chosen; missing artifact error; serving check rejects a missing column and a mostly-null column and accepts a constant column.

### Task 5: Current rosters

**Files:** create `src/serving/rosters.py`; test `tests/test_serving_rosters.py`.

**Interfaces**
- `validate_roster(df)`: needs the four columns, at least 25 distinct teams, no duplicate `PERSON_ID`; raises `DataQualityError` otherwise.
- `fetch_roster(season: str) -> DataFrame`: `nba_api` `CommonAllPlayers(is_only_current_season=1)`, rostered players only (`TEAM_ID != 0`), validated.

**Tests:** `validate_roster` accepts a healthy frame, rejects too few teams, duplicates, and missing columns. The network function is exercised by the real-data run in Task 8.

### Task 6: The slate builder

**Files:** create `src/serving/slate.py`; test `tests/test_serving_slate.py`.

**Interfaces**
- `build_slate(log, roster, history_players, history_teams, models, as_of, slate_date, min_games=5, min_books=3, include_started=False) -> SlateResult`.
- `SlateResult`: `slate: DataFrame`, `skipped: DataFrame (player, event_id, reason)`, `meta: dict (history_last_date, model_trained_through, model_label, n_lines, n_projected, n_skipped)`.
- `slate` columns: `event_id, tip_et, home, away, player, player_id, team, opponent, market, line, books, projection, difference, min_pred, prior_games, captured_at, mins_before_tip`; sorted by `tip_et` then `abs(difference)` descending.
- Skip reasons (exact strings): `unknown team name`, `line from only N book(s)`, `not on either team's roster`, `ambiguous name`, `only N prior games`.
- History is truncated to `GAME_DATE < slate_date`; the freshness gate raises `DataQualityError` when the newest of those games is in the same season as `slate_date` and more than 3 days older.

**Tests (synthetic league + stub models whose predictions are simple functions of features):** projection equals the stub's function of the builder's features (wiring); `difference = projection - line`; rookie with 2 games is skipped with the right reason; unmatched name skipped with reason; 2-book line skipped; a game that already started is excluded unless `include_started`; history after the slate date is ignored; stale history raises; slate sorted as specified; skipped players never appear in the slate.

### Task 7: Command line, rendering, config

**Files:** create the `__main__` entry in `src/serving/slate.py` and `render_slate(result) -> str`; modify `.env.example` (add `LINES_LOG_PATH`), `README.md` (a Live slate section); test `tests/test_serving_slate.py` (render).

**Interfaces:** `python -m src.serving.slate [--date YYYY-MM-DD] [--include-started] [--log-path PATH] [--out PATH]`. The log path defaults to `LINES_LOG_PATH` (from the environment or `.env`). It refreshes the current season's history, loads the newest models and current rosters, builds the slate, prints `render_slate`, and writes `data/processed/slate_<date>.csv`.

**Tests:** `render_slate` shows ET tip time, player, line, projection, difference, and a skipped summary with reasons; contains no EV/edge/BET words.

### Task 8: Real-data run, verification, docs

No new code. Steps: (1) run the command for a recent night with `--include-started` against the real captured log; (2) compare each player's roster team with the team in the actual box score and report agreement; (3) confirm every skipped player has a reason and the counts add up; (4) run the full suite; (5) update the README and commit; (6) finish the branch with the user.

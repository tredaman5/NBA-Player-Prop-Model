# Milestone 5: Daily Workflow and Slate Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After every line capture, GitHub Actions runs the daily pipeline and saves a readable slate page (searchable, sortable) to the private data repo. Nothing from this milestone is public.

**Architecture:** `src/pipeline/page.py` renders the slate to one self-contained HTML file (server-rendered table, a little inline JS for search/sort, no external requests). `run_pipeline` writes it next to the CSV and run summary. A new workflow `daily-slate.yml` fires when the capture workflow finishes, checks out the private data repo (lines and models), runs the pipeline, and commits `slates/` back to the private repo. You read it by pulling your local clone and opening `slates/latest.html`.

**Tech Stack:** pandas, existing pipeline, GitHub Actions. No new Python dependencies.

**Spec:** `design/2026-10-21-slate-rewrite.md` (milestone 5 of 6, narrowed by the owner's choices below).

Commands assume the project virtualenv is active and are run from the repo root. Every code task is test-first.

## Owner decisions (2026-10-24)

- **Private first.** The page contains the real sportsbook lines, so it is committed only to the private repo `nba-prop-data`. No public page this milestone; the Odds API terms bar redistributing their data.
- **Models live in the private data repo** (`models/`), copied there by the owner after local training. The workflow never retrains.

## Global Constraints

- The slate page and every other pipeline output stay out of `docs/` and out of the public repo.
- No "edge", "BET", EV, Kelly, probability or "pick" language on the page. Projection, line, difference only.
- Never write to the owner's local clone of the data repo. The owner copies models into it.
- No external requests from the page (no CDN, fonts or analytics).
- A failed check fails the workflow run; "no games today" (exit code 2) does not.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Known risk

`nba_api` calls stats.nba.com, which often blocks or throttles GitHub's cloud IP ranges. The first manual workflow run is the test. If it is blocked, the fallback is to run the same command on the owner's Mac on a schedule and keep the workflow for the capture only.

## File Structure

- Modify `src/pipeline/daily.py` (`--models-dir`; write the HTML page), `tests/test_pipeline_daily.py`.
- Create `src/pipeline/page.py`, `tests/test_pipeline_page.py`.
- Create `.github/workflows/daily-slate.yml`.
- Modify `.github/workflows/capture-lines.yml` (rebase before push), `README.md`.

---

### Task 1: Models directory option

**Files:** modify `src/pipeline/daily.py`, `tests/test_pipeline_daily.py`.

**Interfaces**
- Produces: `run_pipeline(..., models_dir=None)`: when `load_models_fn` is not given, loads with `load_models(artifacts_dir=models_dir)` if `models_dir` is set, else the default. `main` gets `--models-dir PATH`.

- [ ] **Step 1: Failing test** (append to `tests/test_pipeline_daily.py`)

```python
def test_models_dir_is_passed_to_the_model_loader(monkeypatch, tmp_path):
    seen = {}

    def fake_load_models(artifacts_dir=None, label=None):
        seen["dir"] = artifacts_dir
        return make_models()

    monkeypatch.setattr("src.serving.models.load_models", fake_load_models)
    log_path = tmp_path / "lines.csv"
    make_log().to_csv(log_path, index=False)
    daily.run_pipeline(
        SLATE_DATE, log_path, out_dir=tmp_path / "out", as_of=AS_OF, models_dir=tmp_path / "m",
        load_history_fn=lambda seasons, refresh_seasons=(): make_history(),
        fetch_roster_fn=lambda season: make_roster(),
    )
    assert seen["dir"] == tmp_path / "m"
```

- [ ] **Step 2:** `python -m pytest tests/test_pipeline_daily.py -q`, expect FAIL (`unexpected keyword argument 'models_dir'`).
- [ ] **Step 3: Implement.** In `run_pipeline` add parameter `models_dir=None` after `load_models_fn`; replace `load_models_fn = load_models_fn or load_models` with:

```python
    if load_models_fn is None:
        load_models_fn = (lambda: load_models(artifacts_dir=Path(models_dir))) if models_dir else load_models
```

In `main` add `ap.add_argument("--models-dir", help="folder holding the trained models (default: models_artifacts)")` and pass `models_dir=args.models_dir` to `run_pipeline`.

- [ ] **Step 4:** run the pipeline tests; all pass. **Step 5:** commit `feat(pipeline): choose the models folder`.

### Task 2: The slate page

**Files:** create `src/pipeline/page.py`, `tests/test_pipeline_page.py`; modify `src/pipeline/daily.py`, `tests/test_pipeline_daily.py`.

**Interfaces**
- Produces: `render_page(result: SlateResult, summary: dict) -> str`, a complete HTML document. `run_pipeline` writes it to `out_dir/slate_<date>.html` on success.

Page content: title `NBA slate <date>`; one line of meta (history through, models trained through, generated time in ET); a warning banner for each `summary["warnings"]`; a plain note: "Projection is the model's number and Diff = Projection - Line. There is no injury or lineup information, so large gaps often reflect news the model doesn't have. Not advice."; a search box and a market selector; a table (Tip ET, Game, Player, Team, Mkt, Line, Proj, Diff, Min, Books) rendered server-side in the slate's order, with player and team names HTML-escaped; clicking a header sorts that column in the browser; a collapsed "Skipped" section listing reason, count and names. Neutral styling: no green/red on the difference.

- [ ] **Step 1: Failing tests** `tests/test_pipeline_page.py`

```python
import re

from src.serving.slate import build_slate
from src.pipeline.page import render_page
from tests.slate_fixture import AS_OF, SLATE_DATE, make_history, make_log, make_models, make_roster

SUMMARY = {"warnings": ["models were trained through 2025-01-01, 100 days before this slate; retrain"],
           "as_of": "2025-01-25 12:00:00+00:00"}


def _result(**kw):
    hp, ht = make_history()
    return build_slate(kw.pop("log", make_log()), make_roster(), hp, ht, make_models(),
                       as_of=AS_OF, slate_date=SLATE_DATE, **kw)


def test_the_page_lists_players_with_line_projection_and_difference():
    html = render_page(_result(), SUMMARY)
    assert html.lstrip().lower().startswith("<!doctype html>")
    for text in ("P1", "P4", "10.5", "6:30 PM", "BOS", "DET", "<table"):
        assert text in html


def test_names_are_escaped_so_a_hostile_name_cannot_inject_markup():
    log = make_log(players=("P1", "<script>alert(1)</script>"))
    html = render_page(_result(log=log), SUMMARY)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_warnings_and_skip_reasons_are_shown():
    log = make_log(players=("P1", "P2", "P3", "P4", "Nobody Real"))
    html = render_page(_result(log=log), SUMMARY)
    assert "retrain" in html
    assert "Nobody Real" in html and "not on either team" in html


def test_no_betting_language_and_no_external_requests():
    html = render_page(_result(), SUMMARY)
    assert not re.search(r"\b(edge|bet|ev|kelly|probability|pick)s?\b", html, re.I)
    assert not re.search(r"(src|href)=[\"']https?://", html)


def test_an_empty_slate_still_renders():
    html = render_page(_result(log=make_log(players=("Nobody Real",))), SUMMARY)
    assert "No projections" in html
```

- [ ] **Step 2:** run, expect FAIL (`No module named 'src.pipeline.page'`).
- [ ] **Step 3: Implement `src/pipeline/page.py`.** Build the HTML with f-strings and `html.escape` on every dynamic value. Convert `tip_et` with `strftime("%-I:%M %p")`. Inline `<style>` uses CSS variables on `:root`, a dark-mode override under `@media (prefers-color-scheme: dark)`, system font stack, sticky header row, table wrapped in a horizontally scrollable div, 16px side padding. Inline `<script>` (no external src) wires: the search input filters rows whose text contains the query (case-insensitive); the market select filters on a `data-mkt` attribute; header clicks sort rows by that column (numeric when the cells parse as numbers, toggling direction), and the Diff column sorts by absolute value on first click. The doc must contain none of the banned words; use the wording given above.
- [ ] **Step 4:** In `run_pipeline`, after the summary is complete and before the final `_write_summary`, write `out_dir / f"slate_{slate_date}.html"` with `render_page(result, summary)`. Add to `tests/test_pipeline_daily.py::test_a_good_run_writes_the_slate_and_a_summary` an assertion that `slate_2025-01-25.html` exists.
- [ ] **Step 5:** run the pipeline, page and slate tests; all pass. **Step 6:** commit `feat(pipeline): render the slate as a searchable, sortable page`.

### Task 3: Workflow, setup docs

**Files:** create `.github/workflows/daily-slate.yml`; modify `.github/workflows/capture-lines.yml`, `README.md`.

- [ ] **Step 1: Write `.github/workflows/daily-slate.yml`.** Name `Daily slate`. Triggers: `workflow_run` of `Capture sportsbook lines` (types `completed`), and `workflow_dispatch` with an optional `date` input. `permissions: contents: read`; `concurrency: {group: daily-slate, cancel-in-progress: false}`. One job `slate` on `ubuntu-latest`, `timeout-minutes: 20`, `if: github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success'`. Steps: checkout this repo; setup-python 3.11 with pip cache on `requirements.txt`; `pip install -r requirements.txt` (the pinned versions matter: the model files are pickles); checkout `tredaman5/nba-prop-data` into `data_repo` with `${{ secrets.DATA_REPO_TOKEN }}`; "Verify the data repo is writable" (`git push --dry-run origin HEAD:main`); a step with `id: run` that sets `INPUT_DATE: ${{ inputs.date }}`, builds a bash array `args=(--log-path data_repo/odds_lines_log.csv --models-dir data_repo/models --out-dir out --include-started)`, appends `--date "$INPUT_DATE"` when it is non-empty, runs `python -m src.pipeline.daily "${args[@]}"` with `set +e`, writes `code=$?` to `$GITHUB_OUTPUT`, prints a `::notice::` and exits 0 on code 2, and otherwise exits with that code; then a step `if: always() && steps.run.outputs.code != ''` that copies `out/*.html`, `out/*.json` and `out/*.csv` into `data_repo/slates/`, copies the newest html to `data_repo/slates/latest.html`, and if `git status --porcelain` is non-empty commits as `slate-bot` with message `slate <UTC time>`, runs `git pull --rebase origin main`, and pushes.
- [ ] **Step 2: Edit `capture-lines.yml`.** In "Save the log to the private repo", add `git pull --rebase origin main` immediately before `git push`, so a slate commit that lands first cannot make a capture lose its push (the two workflows touch different files, so the rebase cannot conflict).
- [ ] **Step 3: Validate both YAML files parse:** `ruby -ryaml -e 'ARGV.each { |f| YAML.load_file(f); puts "ok #{f}" }' .github/workflows/*.yml`. Expected: `ok` for each. Re-read the new workflow once for secrets: no secret is echoed; the token is only used by `actions/checkout`.
- [ ] **Step 4: README.** Add a "Daily slate workflow" section: what it does; the one-time setup (copy the models into the data repo with the commands below, nothing else to configure because `DATA_REPO_TOKEN` already exists); how to read the page (`git pull` in the data-repo clone, then open `slates/latest.html`); how to retrain and refresh the models; the nba.com blocking risk and the fallback; and exit codes in a scheduled run.

```bash
mkdir -p ~/Projects/nba-prop-data/models
cp models_artifacts/*_2025-26_RegularSeason.* ~/Projects/nba-prop-data/models/
cd ~/Projects/nba-prop-data && git add models && git commit -m "add trained models" && git push
```

- [ ] **Step 5:** run the full suite. **Step 6:** commit `feat: daily slate workflow that saves the page to the private repo`.

### Task 4: Local verification and finish

No new code.

- [ ] **Step 1:** Run `python -m src.pipeline.daily --log-path <private log> --models-dir models_artifacts --out-dir <scratchpad> --include-started` and open the HTML in the browser pane (desktop width and 375px). Check: the table fills, search and sort work, the warning banner shows, the skipped list matches the CSV counts, the dark scheme is readable, nothing requests an external URL (network panel empty).
- [ ] **Step 2:** Confirm `git status` shows nothing from `out/` or `data/` staged and `grep -rn "slate_" docs/` finds nothing.
- [ ] **Step 3:** Use superpowers:finishing-a-development-branch (merge / PR / keep). Push only on "push". After the push, walk the owner through the one-time model copy and the first manual run of the "Daily slate" workflow, and read the result (including whether nba.com blocked GitHub) from what they paste back.

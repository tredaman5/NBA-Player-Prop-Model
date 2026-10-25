#!/bin/bash
# Builds today's slate on this Mac and saves the page to <data repo>/slates/.
#
# GitHub Actions can't do this: stats.nba.com does not answer GitHub's servers. This runs from
# a macOS scheduled job (scripts/install_schedule.sh) or by hand. It only reads and writes
# your local clone of the private data repo; it never commits or pushes anything.
#
# Settings (environment variables, all optional):
#   NBA_DATA_REPO     your clone of the private data repo   (default ~/Projects/nba-prop-data)
#   NBA_MIN_BATTERY   skip when unplugged and below this %   (default 20)
#   NBA_SLATE_LOG     where output is logged                 (default ~/Library/Logs/nba-slate.log)

set -u
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NBA_DATA_REPO:-$HOME/Projects/nba-prop-data}"
LOG="${NBA_SLATE_LOG:-$HOME/Library/Logs/nba-slate.log}"
MIN_BATTERY="${NBA_MIN_BATTERY:-20}"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=python3

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date) ==="

batt="$(pmset -g batt 2>/dev/null)"
if echo "$batt" | grep -q "Battery Power"; then
  pct="$(echo "$batt" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')"
  if [ -n "$pct" ] && [ "$pct" -lt "$MIN_BATTERY" ]; then
    echo "On battery at ${pct}% (limit ${MIN_BATTERY}%); skipping this run."
    exit 0
  fi
fi

if ! git -C "$DATA" pull --ff-only --quiet; then
  echo "WARNING: could not update the data repo; using the lines it already has."
fi

cd "$REPO" || exit 1
"$PY" -m src.pipeline.daily --include-started \
  --log-path "$DATA/odds_lines_log.csv" \
  --models-dir "$DATA/models" \
  --out-dir "$DATA/slates"
code=$?

if [ "$code" -eq 0 ]; then
  latest="$(ls -t "$DATA"/slates/slate_*.html | head -1)"
  cp "$latest" "$DATA/slates/latest.html"
  echo "OK: $DATA/slates/latest.html"
elif [ "$code" -eq 2 ]; then
  echo "No games or captured lines for today; nothing to build."
else
  echo "FAILED (exit $code)"
  osascript -e 'display notification "The daily slate failed. See ~/Library/Logs/nba-slate.log" with title "NBA slate"' || true
fi
exit "$code"

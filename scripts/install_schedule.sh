#!/bin/bash
# Installs (or removes, with --uninstall) the macOS scheduled job that runs scripts/daily_local.sh
# every day at 6:30 pm local time. A run missed because the Mac was asleep happens
# when it wakes. Run this yourself; nothing else installs it.

set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.tredaman5.nba-slate"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/daily_local.sh</string></array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
  </array>
</dict>
</plist>
PLIST_EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "Installed $LABEL (runs daily at 6:30 pm local time). Log: ~/Library/Logs/nba-slate.log"

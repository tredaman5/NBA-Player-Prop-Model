"""
env.py

Loads secrets from a local .env file so nobody has to export ODDS_API_KEY in
every shell. Copy .env.example to .env and put your key in it. .env is
gitignored; never commit it.

Real environment variables always win over the file, so GitHub Actions (which
injects secrets as environment variables) and a one-off `export` both still
work. No third-party dependency: the format is just KEY=VALUE lines, with
optional `export ` prefix, optional quotes, and full-line # comments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

# What .env.example ships with; treated as "not set" so a forgotten edit gives a
# clear message instead of a confusing 401 from the API.
PLACEHOLDER = "your_key_here"


def load_env_file(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from the .env file into os.environ (existing vars win)."""
    path = path or ENV_FILE
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def require_odds_api_key() -> str:
    """Return ODDS_API_KEY from the environment or .env, or raise with how to fix it."""
    load_env_file()
    key = os.getenv("ODDS_API_KEY")
    if not key or key == PLACEHOLDER:
        raise RuntimeError(
            "Missing ODDS_API_KEY. Copy .env.example to .env and put your key in it "
            "(or export ODDS_API_KEY in your shell)."
        )
    return key

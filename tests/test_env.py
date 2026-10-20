"""
Tests for src/common/env.py: the .env loader that saves people from exporting
ODDS_API_KEY in every shell.

Includes guards that the secret file can't be committed by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.common.env as env
from src.common.env import PLACEHOLDER, load_env_file, require_odds_api_key

REPO_ROOT = Path(__file__).resolve().parents[1]


def _unset(monkeypatch, *names):
    """Make sure these vars are absent now AND removed again at teardown, even when the
    loader (not monkeypatch) is what sets them."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_parses_plain_export_quoted_and_commented_lines(monkeypatch, tmp_path):
    _unset(monkeypatch, "T_PLAIN", "T_EXPORT", "T_DQ", "T_SQ", "T_EQ")
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        "T_PLAIN=abc123\n"
        "export T_EXPORT=exported\n"
        'T_DQ="double quoted"\n'
        "T_SQ='single quoted'\n"
        "T_EQ=a=b=c\n"
        "not a pair\n"
    )
    load_env_file(f)

    assert os.environ["T_PLAIN"] == "abc123"
    assert os.environ["T_EXPORT"] == "exported"
    assert os.environ["T_DQ"] == "double quoted"
    assert os.environ["T_SQ"] == "single quoted"
    assert os.environ["T_EQ"] == "a=b=c"


def test_real_environment_variable_wins_over_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("T_WINS", "from-shell")
    f = tmp_path / ".env"
    f.write_text("T_WINS=from-file\n")
    load_env_file(f)
    assert os.environ["T_WINS"] == "from-shell"  # so Actions secrets and one-off exports still work


def test_missing_file_is_a_no_op(tmp_path):
    load_env_file(tmp_path / "does-not-exist.env")  # must not raise


def test_key_comes_from_the_env_file_when_the_shell_has_none(monkeypatch, tmp_path):
    _unset(monkeypatch, "ODDS_API_KEY")
    f = tmp_path / ".env"
    f.write_text("ODDS_API_KEY=file-key\n")
    monkeypatch.setattr(env, "ENV_FILE", f)
    assert require_odds_api_key() == "file-key"


def test_missing_key_error_tells_you_how_to_fix_it(monkeypatch):
    _unset(monkeypatch, "ODDS_API_KEY")
    with pytest.raises(RuntimeError, match=r"\.env"):
        require_odds_api_key()


def test_unedited_placeholder_counts_as_missing(monkeypatch, tmp_path):
    _unset(monkeypatch, "ODDS_API_KEY")
    f = tmp_path / ".env"
    f.write_text(f"ODDS_API_KEY={PLACEHOLDER}\n")
    monkeypatch.setattr(env, "ENV_FILE", f)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        require_odds_api_key()


def test_env_file_is_gitignored_but_the_example_is_not():
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines, "a real .env could be committed to a public repo"
    assert "!.env.example" in lines


def test_example_file_ships_only_a_placeholder():
    example = (REPO_ROOT / ".env.example").read_text()
    assert f"ODDS_API_KEY={PLACEHOLDER}" in example

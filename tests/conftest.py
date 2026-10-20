import pytest


@pytest.fixture(autouse=True)
def _isolate_env_file(monkeypatch, tmp_path):
    """Point the .env loader at a file that doesn't exist, so a developer's real .env
    (with a real key in it) can never change what a test sees."""
    monkeypatch.setattr("src.common.env.ENV_FILE", tmp_path / "no-such.env")

"""
models.py: load the trained minutes and prop models, with their feature lists and error spread.

Artifacts are written by src/training/train_all.py as {name}_{model|features|eval}_{label}.*
where label is "<latest season>_<season type without spaces>".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models_artifacts"

MARKET_TO_PROP = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists"}
PROPS = ("points", "rebounds", "assists")
_NAMES = ("minutes",) + PROPS
_MODEL_FILE = re.compile(r"^(minutes|points|rebounds|assists)_model_(.+)\.joblib$")


@dataclass
class Models:
    minutes: Any
    props: Dict[str, Any]
    features: Dict[str, List[str]]
    rmse: Dict[str, float]
    trained_through: str
    label: str


def _files(artifacts_dir: Path, label: str) -> List[Path]:
    return [artifacts_dir / f"{n}_{kind}_{label}.{ext}" for n in _NAMES
            for kind, ext in (("model", "joblib"), ("features", "json"), ("eval", "json"))]


def _labels_newest_first(artifacts_dir: Path) -> List[str]:
    found = {m.group(2) for f in artifacts_dir.glob("*_model_*.joblib") if (m := _MODEL_FILE.match(f.name))}
    return sorted(found, reverse=True)


def _load(artifacts_dir: Path, label: str) -> Models:
    def eval_of(name: str) -> dict:
        return json.loads((artifacts_dir / f"{name}_eval_{label}.json").read_text())

    return Models(
        minutes=joblib.load(artifacts_dir / f"minutes_model_{label}.joblib"),
        props={p: joblib.load(artifacts_dir / f"{p}_model_{label}.joblib") for p in PROPS},
        features={n: json.loads((artifacts_dir / f"{n}_features_{label}.json").read_text()) for n in _NAMES},
        rmse={p: float(eval_of(p)["rmse"]) for p in PROPS},
        trained_through=str(eval_of("minutes").get("trained_through", "unknown")),
        label=label,
    )


def load_models(artifacts_dir: Path = ARTIFACTS_DIR, label: Optional[str] = None) -> Models:
    """The newest complete set of models, or the one named by `label` (which must be complete)."""
    artifacts_dir = Path(artifacts_dir)
    if label is not None:
        missing = [f.name for f in _files(artifacts_dir, label) if not f.exists()]
        if missing:
            raise FileNotFoundError(f"missing model artifact(s) for {label}: {missing}")
        return _load(artifacts_dir, label)

    for candidate in _labels_newest_first(artifacts_dir):
        if all(f.exists() for f in _files(artifacts_dir, candidate)):
            return _load(artifacts_dir, candidate)
    raise FileNotFoundError(f"no complete set of model artifacts in {artifacts_dir}")

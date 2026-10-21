"""Feature names, per model. One place, so the builder and the trainers cannot drift apart."""

from __future__ import annotations

MIN_WINDOWS = (3, 5, 10)
RATE_STATS = ("FGA", "FTA", "FG3A", "PTS", "REB", "AST")
TEAM_PTS_WINDOWS = (5, 10)
OPP_STATS = ("PTS", "REB", "AST")
OPP_WINDOWS = (5, 10)
USAGE_WINDOWS = (3, 5, 10)

_CONTEXT = ["IS_HOME", "DAYS_REST", "IS_B2B"]
_MIN_ROLL = [f"MIN_ROLL_{w}" for w in MIN_WINDOWS]
_USAGE = [f"USAGE_RATE_ROLL_{w}" for w in USAGE_WINDOWS]


def _rates(stat: str) -> list[str]:
    return [f"{stat}_PER_MIN_ROLL_{w}" for w in MIN_WINDOWS]


def _opp(stat: str) -> list[str]:
    return [f"OPP_{stat}_ALLOWED_ROLL_{w}" for w in OPP_WINDOWS]


_TEAM_PTS = [f"TEAM_PTS_ROLL_{w}" for w in TEAM_PTS_WINDOWS]

# Everything the builder emits that a model may use.
BUILDER_FEATURES = (
    _CONTEXT
    + _MIN_ROLL
    + [c for s in RATE_STATS for c in _rates(s)]
    + _USAGE
    + _TEAM_PTS
    + [c for s in OPP_STATS for c in _opp(s)]
)

MINUTES_FEATURES = _CONTEXT + _MIN_ROLL

PROP_FEATURES = {
    "points": (
        _CONTEXT + _MIN_ROLL + ["MIN_PRED"]
        + _rates("FGA") + _rates("FTA") + _rates("FG3A") + _rates("PTS")
        + _TEAM_PTS + _opp("PTS") + _USAGE
    ),
    "rebounds": _CONTEXT + _MIN_ROLL + ["MIN_PRED"] + _rates("REB") + _opp("REB") + _USAGE,
    "assists": _CONTEXT + _MIN_ROLL + ["MIN_PRED"] + _rates("AST") + _opp("AST") + _USAGE,
}

PROP_TARGETS = {"points": "PTS", "rebounds": "REB", "assists": "AST"}

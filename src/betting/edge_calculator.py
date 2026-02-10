"""
edge_calculator.py

Betting math utilities:
- American odds <-> implied probability
- Edge & expected value
- Fractional Kelly sizing

This module does NOT scrape odds or place bets.
It takes your model probabilities and sportsbook odds as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Side = Literal["over", "under"]


@dataclass(frozen=True)
class EdgeResult:
    side: Side
    p_model: float
    odds_american: int
    p_implied: float
    edge: float
    ev_per_dollar: float
    kelly_fraction: float
    suggested_fractional_kelly: float


def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.
    Examples:
      -110 -> 1.9091
      +120 -> 2.20
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(odds))


def american_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability (no-vig not removed).
    -110 -> 0.5238
    +120 -> 0.4545
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.
    """
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")

    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    else:
        return int(round(-100.0 / (decimal_odds - 1.0)))


def clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, p))


def expected_value_per_dollar(p_win: float, odds_american: int) -> float:
    """
    Expected profit per $1 staked.
    If EV is 0.03 => +3 cents per $1 long-run (before limits/variance).
    """
    p_win = clamp_prob(p_win)
    dec = american_to_decimal(odds_american)

    profit_if_win = dec - 1.0  # net profit on $1 stake
    loss_if_lose = 1.0

    return p_win * profit_if_win - (1.0 - p_win) * loss_if_lose


def kelly_fraction(p_win: float, odds_american: int) -> float:
    """
    Full Kelly fraction for a single bet.
    f* = (b*p - q) / b, where b = decimal_odds - 1, q = 1-p
    """
    p = clamp_prob(p_win)
    q = 1.0 - p
    b = american_to_decimal(odds_american) - 1.0

    if b <= 0:
        return 0.0

    f = (b * p - q) / b
    return max(0.0, f)


def compute_edge(
    p_over: float,
    odds_over: int,
    odds_under: Optional[int] = None,
    side: Side = "over",
    kelly_multiplier: float = 0.25,
) -> EdgeResult:
    """
    Compute edge + EV + Kelly sizing for OVER or UNDER.

    Inputs:
    - p_over: your model probability that the outcome is OVER the line
    - odds_over: American odds for over
    - odds_under: American odds for under (optional; used if side="under")
    - side: "over" or "under"
    - kelly_multiplier: 0.25 = quarter Kelly (recommended)

    Returns EdgeResult for the chosen side.
    """
    p_over = clamp_prob(p_over)

    if side == "over":
        p_model = p_over
        odds = odds_over
    else:
        if odds_under is None:
            raise ValueError("odds_under is required when side='under'")
        p_model = 1.0 - p_over
        odds = odds_under

    p_impl = american_to_implied_prob(odds)
    edge = p_model - p_impl
    ev = expected_value_per_dollar(p_model, odds)
    kelly = kelly_fraction(p_model, odds)
    frac_kelly = kelly * kelly_multiplier

    return EdgeResult(
        side=side,
        p_model=p_model,
        odds_american=odds,
        p_implied=p_impl,
        edge=edge,
        ev_per_dollar=ev,
        kelly_fraction=kelly,
        suggested_fractional_kelly=frac_kelly,
    )


def should_bet(
    edge_result: EdgeResult,
    min_edge: float = 0.02,
    min_ev: float = 0.0,
) -> bool:
    """
    Simple decision rule:
    - edge >= min_edge (e.g. 2%)
    - EV >= min_ev (e.g. >= 0)
    """
    return (edge_result.edge >= min_edge) and (edge_result.ev_per_dollar >= min_ev)

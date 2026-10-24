"""A tiny deterministic league for feature tests: teams A and B, two players each,
one game every two days, A and B alternating home and away."""

from __future__ import annotations

import pandas as pd

def make_league(
    n_games: int = 12,
    gap_after: int | None = None,
    gap_days: int = 150,
    teams: tuple = ("A", "B"),
):
    """Return (player_games, team_games). If gap_after is set, every game from that index on is
    pushed gap_days later, to mimic an offseason. `teams` names the two teams (ids 100 and 200)."""
    team_a, team_b = teams
    TEAM_IDS = {team_a: 100, team_b: 200}
    start = pd.Timestamp("2025-01-01")
    player_rows, team_rows = [], []
    for g in range(n_games):
        date = start + pd.Timedelta(days=2 * g)
        if gap_after is not None and g >= gap_after:
            date += pd.Timedelta(days=gap_days)
        home, away = (team_a, team_b) if g % 2 == 0 else (team_b, team_a)
        game_id = f"g{g + 1}"
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            matchup = f"{team} vs. {opp}" if is_home else f"{team} @ {opp}"
            totals = dict.fromkeys(["PTS", "REB", "AST", "FGA", "FTA", "TOV"], 0)
            for slot in (0, 1):
                pid = (1 if team == team_a else 3) + slot
                pts = 10 + 2 * g + 3 * slot + (5 if team == team_a else 0) + g % 3
                stats = {
                    "MIN": 32 - 10 * slot + g % 4,
                    "PTS": pts,
                    "REB": 3 + slot + g % 2,
                    "AST": 2 + slot + g % 3,
                    "FGA": pts - 2,
                    "FTA": 2 + g % 3,
                    "FG3A": slot + g % 2,
                    "TOV": 1 + g % 2,
                }
                player_rows.append(
                    {
                        "GAME_ID": game_id, "GAME_DATE": date, "PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}",
                        "TEAM_ID": TEAM_IDS[team], "TEAM_ABBREVIATION": team, "MATCHUP": matchup, **stats,
                    }
                )
                for key in totals:
                    totals[key] += stats[key]
            team_rows.append(
                {
                    "GAME_ID": game_id, "GAME_DATE": date, "TEAM_ID": TEAM_IDS[team],
                    "TEAM_ABBREVIATION": team, "MATCHUP": matchup, **totals,
                }
            )
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)

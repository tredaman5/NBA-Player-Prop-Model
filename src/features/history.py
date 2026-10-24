"""
history.py: one continuous player-game log and team-game log across seasons.

Rolling windows and rest days then carry over the offseason instead of restarting every
October. Every pull is validated, and the combined history is validated again, which also
catches the same game appearing in two seasons.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

from src.common.validation import EmptyDataError, validate_player_gamelog, validate_team_gamelog
from src.features.builder import PLAYER_COLS, TEAM_COLS


def load_history(
    seasons: Sequence[str],
    season_type: str = "Regular Season",
    use_cache: bool = True,
    fetch_players: Optional[Callable] = None,
    fetch_teams: Optional[Callable] = None,
    refresh_seasons: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (player_games, team_games) for all `seasons`, validated and sorted by date.

    refresh_seasons: seasons still being played. They are always fetched fresh (a cache written
    mid-season goes stale), and a refresh season with no games yet (opening day) is skipped
    instead of failing. A finished season that comes back empty is still an error.
    """
    if not seasons:
        raise ValueError("load_history needs at least one season")
    if fetch_players is None:
        from src.minutes.build_dataset import fetch_league_player_gamelogs as fetch_players
    if fetch_teams is None:
        from src.common.opponent_stats import fetch_league_team_gamelogs as fetch_teams

    players, teams = [], []
    for season in seasons:
        refreshing = season in refresh_seasons
        try:
            p = fetch_players(season, season_type=season_type, use_cache=use_cache and not refreshing)
            t = fetch_teams(season, season_type=season_type, use_cache=use_cache and not refreshing)
        except EmptyDataError:
            if refreshing:
                continue  # the season has not produced a game yet
            raise
        players.append(p[PLAYER_COLS])
        teams.append(t[TEAM_COLS])

    player_games = pd.concat(players, ignore_index=True)
    team_games = pd.concat(teams, ignore_index=True)
    for frame in (player_games, team_games):
        frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])

    validate_player_gamelog(player_games, "combined player history")
    validate_team_gamelog(team_games, "combined team history")

    sort_keys = ["GAME_DATE", "GAME_ID"]
    return (
        player_games.sort_values(sort_keys).reset_index(drop=True),
        team_games.sort_values(sort_keys).reset_index(drop=True),
    )

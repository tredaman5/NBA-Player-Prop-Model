import pandas as pd
import pytest

from src.serving.lines import consensus_lines, load_line_log, main_lines

COLUMNS = ["captured_at", "event_id", "commence_time", "home_team", "away_team",
           "bookmaker", "market", "player", "line", "over_price", "under_price"]


def _row(book, player, line, over, under, event="e1", captured="2026-10-21T20:00:00+00:00", market="player_points"):
    return {"captured_at": captured, "event_id": event, "commence_time": "2026-10-21T23:30:00+00:00",
            "home_team": "Boston Celtics", "away_team": "Detroit Pistons", "bookmaker": book,
            "market": market, "player": player, "line": line, "over_price": over, "under_price": under}


def _log(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    return df


def test_the_alternate_line_priced_closest_to_even_is_the_main_line():
    log = _log([
        _row("bovada", "A", 18.5, -300, 240),   # alternate, heavily skewed
        _row("bovada", "A", 24.5, -115, -105),  # the real line
        _row("bovada", "A", 30.5, 250, -330),   # alternate, heavily skewed
    ])
    main = main_lines(log)
    assert len(main) == 1 and main.iloc[0]["line"] == 24.5


def test_a_row_missing_a_price_is_dropped():
    log = _log([_row("bovada", "A", 24.5, None, -105), _row("bovada", "A", 25.5, -110, -110)])
    assert main_lines(log).iloc[0]["line"] == 25.5


def test_consensus_is_the_median_across_books_and_counts_books():
    log = _log([_row(b, "A", line, -110, -110) for b, line in (("dk", 24.5), ("fd", 25.5), ("mgm", 26.5), ("br", 25.5))])
    c = consensus_lines(log).iloc[0]
    assert c["line"] == 25.5 and c["books"] == 4 and not c["thin_market"]


def test_a_player_with_fewer_than_three_books_is_a_thin_market():
    log = _log([_row("dk", "A", 24.5, -110, -110), _row("fd", "A", 25.5, -110, -110)])
    c = consensus_lines(log).iloc[0]
    assert c["books"] == 2 and c["thin_market"]


def test_only_the_latest_capture_of_an_event_is_used():
    early = [_row(b, "A", 20.5, -110, -110, captured="2026-10-21T16:00:00+00:00") for b in ("dk", "fd", "mgm")]
    late = [_row(b, "A", 24.5, -110, -110, captured="2026-10-21T21:00:00+00:00") for b in ("dk", "fd", "mgm")]
    only_early = _row("dk", "B", 10.5, -110, -110, captured="2026-10-21T16:00:00+00:00")
    c = consensus_lines(_log(early + late + [only_early]))
    assert c[c["player"] == "A"].iloc[0]["line"] == 24.5
    assert "B" not in c["player"].tolist()  # not in the latest snapshot


def test_markets_are_kept_apart():
    log = _log([_row(b, "A", line, -110, -110, market=m)
                for b in ("dk", "fd", "mgm") for m, line in (("player_points", 24.5), ("player_rebounds", 7.5))])
    c = consensus_lines(log)
    assert sorted(zip(c["market"], c["line"])) == [("player_points", 24.5), ("player_rebounds", 7.5)]


def test_load_line_log_parses_dates_and_rejects_bad_files(tmp_path):
    good = tmp_path / "log.csv"
    pd.DataFrame([_row("dk", "A", 24.5, -110, -110)]).to_csv(good, index=False)
    log = load_line_log(good)
    assert str(log["commence_time"].dt.tz) == "UTC"

    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=COLUMNS).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_line_log(empty)

    wrong = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1]}).to_csv(wrong, index=False)
    with pytest.raises(ValueError, match="columns"):
        load_line_log(wrong)

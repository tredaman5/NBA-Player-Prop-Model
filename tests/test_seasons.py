from src.common.seasons import season_for_date, season_label


def test_season_label():
    assert season_label(2025) == "2025-26"
    assert season_label(1999) == "1999-00"


def test_season_for_date_covers_october_through_the_finals():
    assert season_for_date("2025-10-21") == "2025-26"
    assert season_for_date("2026-04-12") == "2025-26"
    assert season_for_date("2026-06-15") == "2025-26"
    assert season_for_date("2026-10-20") == "2026-27"

from datetime import datetime, timedelta, timezone

import pytest

from scaleoututil.utils.time import format_iso_utc, parse_iso_utc, to_utc, utc_now

CET = timezone(timedelta(hours=1))


def test_utc_now_is_aware_utc():
    now = utc_now()
    assert now.utcoffset() == timedelta(0)
    assert abs(now - datetime.now(timezone.utc)) < timedelta(seconds=5)


def test_to_utc_converts_offset():
    assert to_utc(datetime(2026, 1, 1, 13, 0, tzinfo=CET)) == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert to_utc(datetime(2026, 1, 1, 13, 0, tzinfo=CET)).tzinfo == timezone.utc


def test_to_utc_reads_naive_as_local_time():
    local = datetime(2026, 1, 1, 12, 0)  # noqa: DTZ001
    assert to_utc(local) == local.astimezone()


@pytest.mark.parametrize(
    "value",
    ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00+00:00", "2026-01-01T13:00:00+01:00", "2026-01-01T12:00:00"],
)
def test_parse_iso_utc(value):
    parsed = parse_iso_utc(value)
    assert parsed == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert parsed.tzinfo == timezone.utc


def test_format_iso_utc():
    assert format_iso_utc(datetime(2026, 1, 1, 13, 0, 0, 123456, tzinfo=CET)) == "2026-01-01T12:00:00.123Z"
    assert format_iso_utc(datetime(2026, 1, 1, 12, 0)) == "2026-01-01T12:00:00.000Z"  # noqa: DTZ001
    assert format_iso_utc(None) is None

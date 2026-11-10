"""Timezone-aware UTC helpers.

All timestamps exchanged between Scaleout components and stored in the database are aware UTC.
"""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to aware UTC.

    A naive datetime is assumed to be local time, as produced by ``datetime.now()``.
    """
    return dt.astimezone(timezone.utc)


def parse_iso_utc(value: str) -> datetime:
    """Parse an ISO 8601 string into an aware UTC datetime.

    Accepts a trailing ``Z``. A string without an offset is read as UTC.
    """
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime as ISO 8601 UTC with milliseconds and ``Z``, e.g. ``2026-10-16T12:00:00.000Z``.

    A naive datetime is read as UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

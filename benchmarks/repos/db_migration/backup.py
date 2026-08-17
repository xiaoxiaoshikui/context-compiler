"""Backup snapshot bookkeeping for the primary database."""

from datetime import datetime, timezone


def latest_backup_age_hours(snapshot_timestamp: datetime) -> float:
    now = datetime.now(timezone.utc)
    return (now - snapshot_timestamp).total_seconds() / 3600.0

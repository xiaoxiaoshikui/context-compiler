"""Apply pending schema migrations in order."""

from pathlib import Path

from .backup import latest_backup_age_hours


def apply_migrations(migrations_dir: Path, *, allow_destructive: bool = False) -> list[str]:
    applied = []
    for path in sorted(migrations_dir.glob("*.sql")):
        sql = path.read_text()
        is_destructive = "DROP TABLE" in sql.upper() or "DROP COLUMN" in sql.upper()
        if is_destructive and not allow_destructive:
            raise RuntimeError(f"{path.name} is destructive; see MIGRATION_POLICY.md")
        applied.append(path.name)
    return applied

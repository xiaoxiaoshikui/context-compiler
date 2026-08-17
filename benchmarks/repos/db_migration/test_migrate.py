import pytest

from db.migrate import apply_migrations


def test_destructive_migration_is_blocked_by_default(tmp_path):
    (tmp_path / "0001_drop_sessions_v1.sql").write_text("DROP TABLE sessions_v1;")
    with pytest.raises(RuntimeError):
        apply_migrations(tmp_path)

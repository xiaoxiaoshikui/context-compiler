# Migration safety policy

Never run a destructive schema migration (DROP TABLE, DROP COLUMN, or any
irreversible data loss) against the production database without a verified
backup snapshot that is less than 24 hours old.

Before running a destructive migration:

1. Confirm the latest backup snapshot age via backup.latest_backup_age_hours.
2. Get sign-off from the on-call database owner.
3. Run the migration with --allow-destructive only after both are satisfied.

This applies even to migrations that only affect tables believed to be
unused, such as legacy `_v1` tables.

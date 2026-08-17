# Schema Migrations

Utilities for applying versioned schema migrations to the primary Postgres
database. See MIGRATION_POLICY.md before writing any migration that drops
or alters a table used in production traffic.

Migrations live in db/migrations and are applied with `python -m db.migrate`.

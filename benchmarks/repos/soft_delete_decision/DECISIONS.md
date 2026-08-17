# Architecture decisions

## Customer records use soft delete

We use soft delete (a `deleted_at` column) instead of a hard SQL DELETE
for customer records. Compliance requires us to produce a full audit
trail of account changes for 7 years, even after a user requests deletion
of their active account. A hard delete would make that audit trail
impossible to reconstruct.

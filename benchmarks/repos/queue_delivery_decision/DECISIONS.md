# Architecture decisions

## Queue delivery semantics

We deliberately use at-least-once delivery with idempotent consumers,
rather than at-most-once delivery, for the order events queue. Losing a
single payment-confirmed event silently is worse than processing one
event twice when every consumer is written to be idempotent (keyed on
`event_id`). Do not switch this queue to at-most-once delivery to
simplify consumer code; instead keep consumers idempotent.

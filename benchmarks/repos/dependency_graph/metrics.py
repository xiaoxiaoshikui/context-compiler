"""Refund volume counters, unrelated to amount validation."""

_COUNTERS: dict[str, int] = {}


def record_refund(order_id: str) -> None:
    _COUNTERS[order_id] = _COUNTERS.get(order_id, 0) + 1

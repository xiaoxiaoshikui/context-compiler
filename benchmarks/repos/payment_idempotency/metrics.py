"""Prometheus-style counters for the payments service."""

_COUNTERS: dict[str, int] = {}


def increment(name: str, value: int = 1) -> None:
    _COUNTERS[name] = _COUNTERS.get(name, 0) + value


def snapshot() -> dict[str, int]:
    return dict(_COUNTERS)


def reset() -> None:
    _COUNTERS.clear()

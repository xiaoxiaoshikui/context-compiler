"""Latency histograms, unrelated to the backoff schedule itself."""

_HISTOGRAM: list[float] = []


def record_latency(seconds: float) -> None:
    _HISTOGRAM.append(seconds)

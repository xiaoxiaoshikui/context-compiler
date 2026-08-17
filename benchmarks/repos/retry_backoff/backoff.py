"""Exponential backoff retry wrapper."""

import time

from .errors import TransientError


def call_with_backoff(fn, max_attempts=5, base_delay_seconds=0.5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay_seconds * (2**attempt))

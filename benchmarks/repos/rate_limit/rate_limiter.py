"""Per-IP token bucket rate limiter."""

import time


class RateLimiter:
    def __init__(self, capacity: int, refill_per_second: float) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client_ip: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(client_ip, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
        if tokens < 1:
            self._buckets[client_ip] = (tokens, now)
            return False
        self._buckets[client_ip] = (tokens - 1, now)
        return True

"""Small in-process TTL cache."""

import time


class Cache:
    def __init__(self, ttl_seconds=30):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[object, float]] = {}

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (value, time.monotonic() + self.ttl_seconds)

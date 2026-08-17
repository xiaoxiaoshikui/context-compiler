"""Generic retry helper, unrelated to any specific timeout value."""

from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(fn: Callable[[], T], attempts: int = 3) -> T:
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    assert last_exc is not None
    raise last_exc

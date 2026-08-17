"""HTTP handlers that sit behind the rate limiter middleware."""

from .rate_limiter import RateLimiter

_limiter = RateLimiter(capacity=60, refill_per_second=1.0)


def handle_request(client_ip: str, path: str) -> tuple[int, str]:
    if not _limiter.allow(client_ip):
        return 429, "rate limited"
    return 200, "ok"

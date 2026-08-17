from api.rate_limiter import RateLimiter


def test_bucket_blocks_after_capacity_exhausted():
    limiter = RateLimiter(capacity=2, refill_per_second=0.0)
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False

"""Thin HTTP client for the upstream payment gateway."""


class GatewayTimeout(Exception):
    pass


class GatewayClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def capture(self, idempotency_key: str, amount_cents: int, currency: str) -> dict:
        raise NotImplementedError("network call stubbed for tests")

    def lookup_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        raise NotImplementedError("network call stubbed for tests")

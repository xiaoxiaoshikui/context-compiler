"""Thin HTTP client for the billing service."""

from .retry import with_retries


class BillingClient:
    def __init__(self, base_url: str, config: dict) -> None:
        self.base_url = base_url
        self.config = config

    def get_invoice(self, invoice_id: str) -> dict:
        return with_retries(lambda: self._request(f"/invoices/{invoice_id}"))

    def _request(self, path: str) -> dict:
        raise NotImplementedError("network call stubbed for tests")

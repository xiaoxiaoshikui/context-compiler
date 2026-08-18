"""Talks to the payment processor payout API."""


def submit_refund(order_id: str, amount_cents: int) -> dict:
    raise NotImplementedError("payment processor call stubbed for tests")

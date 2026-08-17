"""Charge orchestration against the upstream payment gateway."""

from dataclasses import dataclass

from .gateway_client import GatewayClient, GatewayTimeout


@dataclass
class ChargeRequest:
    customer_id: str
    amount_cents: int
    currency: str
    idempotency_key: str


def charge_customer(client: GatewayClient, request: ChargeRequest) -> dict:
    """Authorize and capture a charge.

    On a gateway timeout the caller does not know whether the charge was
    actually created upstream. Retrying blindly can create a duplicate
    charge, so any retry must reuse the same confirmed idempotency key and
    must first check charge status before resubmitting.
    """
    try:
        return client.capture(request.idempotency_key, request.amount_cents, request.currency)
    except GatewayTimeout:
        status = client.lookup_by_idempotency_key(request.idempotency_key)
        if status is not None:
            return status
        raise

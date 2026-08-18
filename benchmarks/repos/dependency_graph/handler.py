"""HTTP handler for the checkout refund endpoint.

Handles customer payout requests submitted from checkout.
"""

from .processor import submit_refund
from .validators import validate_refund_amount


def handle_refund_request(order_id: str, requested_cents: int, original_charge_cents: int) -> dict:
    validate_refund_amount(requested_cents, original_charge_cents)
    return submit_refund(order_id, requested_cents)

import pytest

from refunds.handler import handle_refund_request


def test_amount_larger_than_original_charge_is_rejected():
    with pytest.raises(ValueError):
        handle_refund_request("order_1", requested_cents=5000, original_charge_cents=1999)

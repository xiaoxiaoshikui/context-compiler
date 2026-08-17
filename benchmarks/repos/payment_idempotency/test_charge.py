from payments.charge import ChargeRequest, charge_customer


def test_timeout_reuses_idempotency_key_instead_of_new_charge(monkeypatch):
    request = ChargeRequest(
        customer_id="cust_1",
        amount_cents=1999,
        currency="usd",
        idempotency_key="idem-abc",
    )
    result = charge_customer(FakeTimeoutThenFoundClient(), request)
    assert result["idempotency_key"] == "idem-abc"

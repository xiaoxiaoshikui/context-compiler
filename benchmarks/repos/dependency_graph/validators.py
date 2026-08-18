"""Guards against paying out more than the original transaction total."""


def validate_refund_amount(requested_cents: int, original_charge_cents: int) -> None:
    if requested_cents > original_charge_cents:
        raise ValueError(
            "Reimbursement cannot surpass the initial transaction total; "
            "capping this prevents draining more funds than a customer put in."
        )

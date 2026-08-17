# Payments Service

Internal service that authorizes and captures customer card charges through
the upstream payment gateway. Owned by the Payments team. See
PAYMENT_POLICY.md for the operational invariants that must never be
violated when handling gateway timeouts or partial failures.

Local development:

1. `pip install -r requirements.txt`
2. `python -m payments.server`

Metrics are exported to the shared observability stack; see metrics.py.

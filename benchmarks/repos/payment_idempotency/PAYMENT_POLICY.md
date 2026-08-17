# Payment retry policy

Never retry a charge without an idempotency key that has already been
confirmed against the gateway. On a gateway timeout, the outcome of the
original request is unknown; retrying with a new charge attempt can bill
the customer twice.

Required behavior on timeout:

1. Do not create a new charge.
2. Look up the existing charge by the same idempotency key.
3. Only resubmit if the gateway confirms the key was never received.

This invariant applies to every code path that calls the gateway, including
manual support tooling.

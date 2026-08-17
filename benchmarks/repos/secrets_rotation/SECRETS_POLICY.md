# Secrets handling policy

Never commit rotated credentials to source control, and never hardcode a
fallback API key in application code. Rotate credentials through the
secrets manager and redeploy the affected service so it picks up the new
value at startup.

If a key must be rotated urgently:

1. Generate the new key in the secrets manager.
2. Redeploy the service so `fetch_secret` returns the new value.
3. Revoke the old key only after the redeploy is confirmed healthy.

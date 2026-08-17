# Rate limiting policy

Never disable the per-IP rate limiter in production, and never raise its
capacity above documented thresholds as a quick fix. It is the last line of
defense against credential-stuffing attacks against customer accounts.

If legitimate users are being rate-limited during a traffic spike, first
verify whether the traffic is a genuine spike or a credential-stuffing
attack pattern before changing limiter configuration. Prefer adding
per-account allowlisting over weakening the global limiter.

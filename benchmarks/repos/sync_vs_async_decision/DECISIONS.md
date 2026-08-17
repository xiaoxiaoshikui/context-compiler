# Architecture decisions

## Email sending is asynchronous

We deliberately send email asynchronously via a background queue instead
of synchronously inside the request handler. SMTP calls have historically
caused request-handler thread pool exhaustion during provider outages,
which took down unrelated endpoints on the same process. Moving email
sending back into the synchronous request path for lower latency would
reintroduce that failure mode.

class OAuthError(RuntimeError):
    pass


def exchange_code(code: str, state: str) -> str:
    if not code:
        raise OAuthError("empty authorization code")
    return f"session:{state}:{code}"

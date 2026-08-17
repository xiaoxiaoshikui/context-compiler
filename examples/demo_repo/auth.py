from oauth_client import exchange_code


def handle_callback(browser: str, code: str, state: str) -> dict:
    """Exchange the OAuth callback code for a session."""
    if not state:
        raise ValueError("missing state")
    token = exchange_code(code=code, state=state)
    return {"browser": browser, "token": token}

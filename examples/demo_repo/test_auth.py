from auth import handle_callback


def test_safari_callback_uses_single_exchange():
    result = handle_callback("Safari", "abc", "state-1")
    assert result["token"].startswith("session:")

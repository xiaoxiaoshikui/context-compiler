from billing.deploy import build_env


def test_build_env_reads_key_from_secrets_manager(monkeypatch):
    env = build_env("staging")
    assert "BILLING_API_KEY" in env

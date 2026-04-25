import pytest

from src.auth.exceptions import AuthTokenError, AuthUnauthorizedError
from src.auth.oauth import GitHubUser
from src.auth.service import AuthService
from src.auth.whitelist import Whitelist


def test_whitelist_is_case_insensitive() -> None:
    whitelist = Whitelist(["JuyaoHuang"])

    assert whitelist.is_allowed("juyaohuang")
    assert whitelist.is_allowed("JUYAOHUANG")
    assert not whitelist.is_allowed("someone-else")


def test_disabled_auth_accepts_default_user_without_token() -> None:
    service = AuthService({"enabled": False})

    user = service.authenticate_bearer_token(None)

    assert user.username == "default"


def test_enabled_auth_creates_and_verifies_token() -> None:
    service = AuthService(
        {
            "enabled": True,
            "jwt": {"secret_key": "test-secret", "algorithm": "HS256", "expire_days": 7},
            "github": {
                "client_id": "client",
                "client_secret": "secret",
                "callback_url": "http://localhost:8430/api/auth/callback",
            },
            "whitelist": {"users": ["JuyaoHuang"]},
        }
    )

    token = service.create_token_for_github_user(
        GitHubUser(username="JuyaoHuang", avatar_url="https://avatar")
    )
    user = service.authenticate_bearer_token(f"Bearer {token}")

    assert user.username == "JuyaoHuang"
    assert user.avatar_url == "https://avatar"


def test_enabled_auth_requires_bearer_token() -> None:
    service = AuthService(
        {
            "enabled": True,
            "jwt": {"secret_key": "test-secret", "algorithm": "HS256", "expire_days": 7},
            "github": {
                "client_id": "client",
                "client_secret": "secret",
                "callback_url": "http://localhost:8430/api/auth/callback",
            },
            "whitelist": {"users": ["JuyaoHuang"]},
        }
    )

    with pytest.raises(AuthTokenError, match="Missing bearer token"):
        service.authenticate_bearer_token(None)


def test_enabled_auth_rejects_non_whitelisted_token_subject() -> None:
    service = AuthService(
        {
            "enabled": True,
            "jwt": {"secret_key": "test-secret", "algorithm": "HS256", "expire_days": 7},
            "github": {
                "client_id": "client",
                "client_secret": "secret",
                "callback_url": "http://localhost:8430/api/auth/callback",
            },
            "whitelist": {"users": ["JuyaoHuang"]},
        }
    )
    assert service.jwt_manager is not None
    token = service.jwt_manager.create_token("someone-else")

    with pytest.raises(AuthUnauthorizedError, match="not whitelisted"):
        service.authenticate_token(token)

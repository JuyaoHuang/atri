"""Tests for Phase 11 authentication route wiring."""

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.auth.oauth import GitHubUser
from src.auth.session import SESSION_COOKIE_NAME


def _base_config(auth_config: dict) -> dict:
    return {
        "server": {"cors": {"enabled": False}},
        "auth": auth_config,
        "storage": {"mode": "json", "json": {"base_path": "data/chats"}},
        "llm": {},
        "memory": {},
    }


def _enabled_auth_config() -> dict:
    return {
        "enabled": True,
        "jwt": {"secret_key": "test-secret", "algorithm": "HS256", "expire_days": 7},
        "github": {
            "client_id": "github-client",
            "client_secret": "github-secret",
            "callback_url": "http://localhost:8430/api/auth/callback",
            "scope": "read:user",
        },
        "frontend": {
            "callback_url": "http://localhost:5173/auth/callback",
            "login_url": "http://localhost:5173/login",
        },
        "whitelist": {"users": ["JuyaoHuang"]},
    }


@pytest.mark.asyncio
async def test_auth_status_disabled() -> None:
    app = create_app(_base_config({"enabled": False}))

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/auth/status")

    assert response.status_code == 200
    assert response.json() == {"enabled": False}


@pytest.mark.asyncio
async def test_auth_me_returns_default_user_when_disabled() -> None:
    app = create_app(_base_config({"enabled": False}))

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "username": "default",
        "avatar_url": None,
        "name": None,
        "auth_enabled": False,
    }


@pytest.mark.asyncio
async def test_auth_login_returns_github_authorization_url_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    monkeypatch.setattr("src.routes.auth.secrets.token_urlsafe", lambda _size: "server-state")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/auth/login?state=nonce")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True

    authorization_url = data["authorization_url"]
    parsed = urlparse(authorization_url)
    params = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/login/oauth/authorize"
    assert params["client_id"] == ["github-client"]
    assert params["redirect_uri"] == ["http://localhost:8430/api/auth/callback"]
    assert params["scope"] == ["read:user"]
    assert params["state"] == ["server-state"]

    set_cookie = response.headers["set-cookie"]
    assert "atri_oauth_state=server-state" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Max-Age=600" in set_cookie
    assert "Path=/api/auth" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Secure" not in set_cookie


@pytest.mark.asyncio
async def test_auth_login_sets_secure_oauth_state_cookie_for_https_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_config = _enabled_auth_config()
    auth_config["github"]["callback_url"] = "https://example.com/api/auth/callback"
    app = create_app(_base_config(auth_config))
    monkeypatch.setattr("src.routes.auth.secrets.token_urlsafe", lambda _size: "server-state")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/auth/login")

    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]


@pytest.mark.asyncio
async def test_auth_callback_rejects_missing_oauth_state() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        response = await client.get("/api/auth/callback?code=github-code")

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    assert location.path == "/auth/callback"
    assert parse_qs(location.query)["error"] == ["invalid_state"]
    app.state.auth_service.github_oauth.exchange_code_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_callback_rejects_mismatched_oauth_state() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        client.cookies.set(
            "atri_oauth_state",
            "expected-state",
            path="/api/auth",
        )
        response = await client.get(
            "/api/auth/callback?code=github-code&state=actual-state",
        )

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    assert parse_qs(location.query)["error"] == ["invalid_state"]
    assert "atri_oauth_state=" in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]
    app.state.auth_service.github_oauth.exchange_code_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_callback_skips_invalid_state_when_session_cookie_is_valid() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock()
    token = app.state.auth_service.jwt_manager.create_token("JuyaoHuang")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token, path="/")
        client.cookies.set(
            "atri_oauth_state",
            "expected-state",
            path="/api/auth",
        )
        response = await client.get(
            "/api/auth/callback?code=github-code&state=actual-state",
        )

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    params = parse_qs(location.query)
    assert location.path == "/auth/callback"
    assert params["success"] == ["1"]
    assert "error" not in params
    assert "atri_oauth_state=" in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]
    app.state.auth_service.github_oauth.exchange_code_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_callback_rejects_invalid_state_when_session_cookie_is_invalid() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        client.cookies.set(SESSION_COOKIE_NAME, "not-a-valid-token", path="/")
        client.cookies.set(
            "atri_oauth_state",
            "expected-state",
            path="/api/auth",
        )
        response = await client.get(
            "/api/auth/callback?code=github-code&state=actual-state",
        )

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    assert parse_qs(location.query)["error"] == ["invalid_state"]
    app.state.auth_service.github_oauth.exchange_code_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_callback_accepts_matching_oauth_state() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock(
        return_value="github-access-token"
    )
    app.state.auth_service.github_oauth.get_user_info = AsyncMock(
        return_value=GitHubUser(
            username="JuyaoHuang",
            avatar_url="https://avatar.example/me.png",
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        client.cookies.set(
            "atri_oauth_state",
            "expected-state",
            path="/api/auth",
        )
        response = await client.get(
            "/api/auth/callback?code=github-code&state=expected-state",
        )

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    params = parse_qs(location.query)
    assert location.path == "/auth/callback"
    assert params["success"] == ["1"]
    assert "token" not in params

    set_cookie = "\n".join(response.headers.get_list("set-cookie"))
    assert "atri_oauth_state=" in set_cookie
    assert f"{SESSION_COOKIE_NAME}=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Max-Age=604800" in set_cookie
    assert "Path=/" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Secure" not in set_cookie
    app.state.auth_service.github_oauth.exchange_code_for_token.assert_awaited_once_with(
        "github-code"
    )
    app.state.auth_service.github_oauth.get_user_info.assert_awaited_once_with(
        "github-access-token"
    )


@pytest.mark.asyncio
async def test_auth_callback_sets_secure_session_cookie_for_https_frontend() -> None:
    auth_config = _enabled_auth_config()
    auth_config["frontend"]["callback_url"] = "https://example.com/auth/callback"
    app = create_app(_base_config(auth_config))
    app.state.auth_service.github_oauth.exchange_code_for_token = AsyncMock(
        return_value="github-access-token"
    )
    app.state.auth_service.github_oauth.get_user_info = AsyncMock(
        return_value=GitHubUser(username="JuyaoHuang")
    )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        client.cookies.set(
            "atri_oauth_state",
            "expected-state",
            path="/api/auth",
        )
        response = await client.get(
            "/api/auth/callback?code=github-code&state=expected-state",
        )

    assert response.status_code == 307
    set_cookie = "\n".join(response.headers.get_list("set-cookie"))
    session_cookie = next(
        cookie for cookie in response.headers.get_list("set-cookie")
        if cookie.startswith(f"{SESSION_COOKIE_NAME}=")
    )
    assert "Secure" in session_cookie
    assert f"{SESSION_COOKIE_NAME}=" in set_cookie


@pytest.mark.asyncio
async def test_auth_me_requires_token_when_enabled() -> None:
    app = create_app(_base_config(_enabled_auth_config()))

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing session cookie"


@pytest.mark.asyncio
async def test_auth_me_returns_cookie_user_when_enabled() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    token = app.state.auth_service.jwt_manager.create_token(
        "JuyaoHuang",
        avatar_url="https://avatar.example/me.png",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token, path="/")
        response = await client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "username": "JuyaoHuang",
        "avatar_url": "https://avatar.example/me.png",
        "name": None,
        "auth_enabled": True,
    }


@pytest.mark.asyncio
async def test_auth_me_still_accepts_bearer_token_for_compatibility() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    token = app.state.auth_service.jwt_manager.create_token("JuyaoHuang")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["username"] == "JuyaoHuang"


@pytest.mark.asyncio
async def test_auth_logout_clears_session_cookie() -> None:
    app = create_app(_base_config(_enabled_auth_config()))

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/auth/logout")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    set_cookie = response.headers["set-cookie"]
    assert f"{SESSION_COOKIE_NAME}=" in set_cookie
    assert "Max-Age=0" in set_cookie


@pytest.mark.asyncio
async def test_auth_middleware_rejects_protected_routes_without_token() -> None:
    app = create_app(_base_config(_enabled_auth_config()))

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/chats")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing session cookie"


@pytest.mark.asyncio
async def test_protected_chat_routes_use_authenticated_user() -> None:
    app = create_app(_base_config(_enabled_auth_config()))
    app.state.storage = AsyncMock()
    app.state.storage.list_chats.return_value = []
    token = app.state.auth_service.jwt_manager.create_token("JuyaoHuang")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token, path="/")
        response = await client.get("/api/chats")

    assert response.status_code == 200
    assert response.json() == []
    app.state.storage.list_chats.assert_awaited_once_with("JuyaoHuang", None)

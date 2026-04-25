"""GitHub OAuth client."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from src.auth.exceptions import AuthConfigError, AuthUnauthorizedError

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


@dataclass(frozen=True)
class GitHubUser:
    """Authenticated GitHub user profile."""

    username: str
    avatar_url: str | None = None
    name: str | None = None


class GitHubOAuth:
    """Minimal GitHub OAuth web flow client."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        callback_url: str,
        scope: str = "read:user",
    ) -> None:
        if not client_id or client_id.startswith("${"):
            raise AuthConfigError("GitHub OAuth client_id is not configured")
        if not client_secret or client_secret.startswith("${"):
            raise AuthConfigError("GitHub OAuth client_secret is not configured")
        if not callback_url:
            raise AuthConfigError("GitHub OAuth callback_url is not configured")
        self.client_id = client_id
        self.client_secret = client_secret
        self.callback_url = callback_url
        self.scope = scope

    def get_authorization_url(self, state: str | None = None) -> str:
        query = {
            "client_id": self.client_id,
            "redirect_uri": self.callback_url,
            "scope": self.scope,
        }
        if state:
            query["state"] = state
        return f"{GITHUB_AUTHORIZE_URL}?{urlencode(query)}"

    async def exchange_code_for_token(self, code: str) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                GITHUB_TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.callback_url,
                },
            )
        response.raise_for_status()
        data = response.json()
        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AuthUnauthorizedError("GitHub did not return an access token")
        return access_token

    async def get_user_info(self, access_token: str) -> GitHubUser:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                GITHUB_USER_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {access_token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        response.raise_for_status()
        data = response.json()
        login = data.get("login")
        if not isinstance(login, str) or not login:
            raise AuthUnauthorizedError("GitHub user profile is missing login")
        avatar_url = data.get("avatar_url")
        name = data.get("name")
        return GitHubUser(
            username=login,
            avatar_url=avatar_url if isinstance(avatar_url, str) else None,
            name=name if isinstance(name, str) else None,
        )


"""GitHub username whitelist checks."""

from __future__ import annotations


class Whitelist:
    """Case-insensitive GitHub username whitelist."""

    def __init__(self, users: list[str] | None = None) -> None:
        self._users = {user.casefold() for user in users or [] if user.strip()}

    @property
    def users(self) -> list[str]:
        return sorted(self._users)

    def is_allowed(self, username: str) -> bool:
        return username.casefold() in self._users


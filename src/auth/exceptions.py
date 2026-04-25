"""Authentication exception hierarchy."""


class AuthError(Exception):
    """Base authentication error."""


class AuthConfigError(AuthError):
    """Authentication configuration is missing or invalid."""


class AuthUnauthorizedError(AuthError):
    """Request is not authenticated or the user is not allowed."""


class AuthTokenError(AuthUnauthorizedError):
    """JWT token is missing, invalid, or expired."""


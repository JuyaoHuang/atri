"""Generate a strong JWT secret for config/auth.yaml.

Usage:
    uv run python scripts/generate_jwt_secret.py
    uv run python scripts/generate_jwt_secret.py --raw
"""

from __future__ import annotations

import argparse
import secrets

DEFAULT_BYTES = 64


def generate_secret(byte_count: int) -> str:
    """Return a URL-safe random secret."""
    if byte_count < 32:
        raise ValueError("byte count must be at least 32")
    return secrets.token_urlsafe(byte_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a strong JWT_SECRET_KEY value for atri/.env.",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=DEFAULT_BYTES,
        help=f"number of random bytes to use; default: {DEFAULT_BYTES}",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print only the secret value, without the JWT_SECRET_KEY= prefix",
    )
    args = parser.parse_args()

    secret = generate_secret(args.bytes)
    if args.raw:
        print(secret)
    else:
        print(f"JWT_SECRET_KEY={secret}")


if __name__ == "__main__":
    main()

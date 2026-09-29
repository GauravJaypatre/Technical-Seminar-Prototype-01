"""User authentication and profile management service."""

from __future__ import annotations

from typing import Any


class UserDatabase:
    """Manages registered users and authentication records."""

    def __init__(self) -> None:
        self.users: dict[str, dict[str, Any]] = {}

    def register_user(self, email: str, username: str) -> dict[str, Any]:
        """Registers a user profile keyed by their email address."""
        profile = {"email": email, "username": username, "active": True}
        self.users[email] = profile
        return profile

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Looks up a user record by email.

        BUG: Email lookup is case-sensitive, causing login failures when
        users enter their email with different casing (e.g. 'User@example.com'
        vs 'user@example.com').
        """
        return self.users.get(email)

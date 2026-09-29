"""Unit tests verifying user registration and lookup."""

from src.auth import UserDatabase


def test_standard_user_lookup():
    db = UserDatabase()
    db.register_user("alice@example.com", "alice")
    user = db.get_user_by_email("alice@example.com")
    assert user is not None
    assert user["username"] == "alice"


def test_case_insensitive_email_lookup():
    db = UserDatabase()
    db.register_user("Alice.Smith@Example.COM", "alice_smith")
    # This must find the user regardless of query casing
    user = db.get_user_by_email("alice.smith@example.com")
    assert user is not None
    assert user["username"] == "alice_smith"

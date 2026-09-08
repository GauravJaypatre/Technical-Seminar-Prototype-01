import pytest
from src.dispatcher import EventDispatcher, DispatchAggregateError

def test_all_listeners_called_when_first_fails():
    dispatcher = EventDispatcher()
    called = []

    def bad_listener(data):
        called.append("bad")
        raise ValueError("Something went wrong")

    def good_listener(data):
        called.append("good")

    dispatcher.subscribe("user_signup", bad_listener)
    dispatcher.subscribe("user_signup", good_listener)

    # FAILS ON UNPATCHED CODE
    with pytest.raises(DispatchAggregateError) as exc_info:
        dispatcher.dispatch("user_signup", {"username": "alice"})

    assert "bad" in called
    assert "good" in called
    assert len(exc_info.value.errors) == 1

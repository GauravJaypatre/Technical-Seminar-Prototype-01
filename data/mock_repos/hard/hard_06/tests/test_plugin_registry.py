import pytest
from src.registry import PluginRegistry

def test_plugin_isolation():
    registry = PluginRegistry()
    p1 = registry.register("AuthPlugin")
    p2 = registry.register("LoggingPlugin")

    p1.set_config("token", "secret123")

    # FAILS ON UNPATCHED CODE
    assert "token" not in p2.metadata
    assert p2.metadata == {}

import pytest
from src.analyzer import ScopeAnalyzer

def test_function_scoping():
    analyzer = ScopeAnalyzer()
    analyzer.declare_variable("global_var", "int")
    analyzer.enter_function()
    analyzer.declare_variable("local_var", "str")
    assert analyzer.lookup("local_var") == "str"
    assert analyzer.lookup("global_var") == "int"
    analyzer.exit_function()
    assert analyzer.lookup("local_var") is None

def test_comprehension_does_not_shadow_outer_var():
    # FAILS ON UNPATCHED CODE
    analyzer = ScopeAnalyzer()
    analyzer.enter_function()
    analyzer.declare_variable("item", "CustomObject")
    
    analyzer.enter_comprehension()
    analyzer.declare_variable("item", "int")
    assert analyzer.lookup("item") == "int"
    analyzer.exit_comprehension()

    assert analyzer.lookup("item") == "CustomObject"

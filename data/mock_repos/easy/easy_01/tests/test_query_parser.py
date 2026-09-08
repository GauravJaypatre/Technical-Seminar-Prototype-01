import pytest
from src.query_parser import parse_query_string

def test_basic_parsing():
    res = parse_query_string("foo=bar&baz=qux")
    assert res == {"foo": "bar", "baz": "qux"}

def test_leading_question_mark():
    res = parse_query_string("?name=alice&age=30")
    assert res == {"name": "alice", "age": "30"}

def test_empty_string():
    assert parse_query_string("") == {}

def test_empty_value_retained():
    # FAILS ON UNPATCHED CODE
    res = parse_query_string("key=&other=val")
    assert "key" in res
    assert res["key"] == ""
    assert res["other"] == "val"

def test_encoded_spaces():
    res = parse_query_string("msg=hello+world&mode=fast")
    assert res["msg"] == "hello world"

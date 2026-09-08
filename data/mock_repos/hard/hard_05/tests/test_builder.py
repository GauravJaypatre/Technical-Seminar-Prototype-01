import pytest
from src.models import Table
from src.builder import QueryBuilder

def test_query_without_alias():
    t = Table("orders")
    qb = QueryBuilder(t).where("status", "completed")
    assert qb.to_sql() == "SELECT * FROM orders WHERE orders.status = 'completed'"

def test_query_with_alias():
    # FAILS ON UNPATCHED CODE
    t = Table("users", alias="u")
    qb = QueryBuilder(t).where("id", "42")
    assert qb.to_sql() == "SELECT * FROM users AS u WHERE u.id = '42'"

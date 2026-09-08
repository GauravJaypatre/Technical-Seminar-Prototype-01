from src.models import Table

class QueryBuilder:
    def __init__(self, table: Table):
        self.table = table
        self.conditions = []

    def where(self, column: str, value: str):
        # BUG: uses table.name unconditionally rather than table.get_qualifier()
        self.conditions.append(f"{self.table.name}.{column} = '{value}'")
        return self

    def to_sql(self) -> str:
        base = f"SELECT * FROM {self.table.name}"
        if self.table.alias:
            base += f" AS {self.table.alias}"
        if self.conditions:
            base += " WHERE " + " AND ".join(self.conditions)
        return base

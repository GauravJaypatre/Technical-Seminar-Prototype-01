class Table:
    def __init__(self, name: str, alias: str = None):
        self.name = name
        self.alias = alias

    def get_qualifier(self) -> str:
        return self.alias if self.alias else self.name

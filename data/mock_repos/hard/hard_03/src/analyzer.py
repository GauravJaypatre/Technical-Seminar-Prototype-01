from src.scope import SymbolTable

class ScopeAnalyzer:
    def __init__(self):
        self.current_scope = SymbolTable()

    def enter_function(self):
        self.current_scope = SymbolTable(parent=self.current_scope)

    def exit_function(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    # BUG: enter_comprehension does not create a child scope
    def enter_comprehension(self):
        pass

    def exit_comprehension(self):
        pass

    def declare_variable(self, name: str, var_type: str):
        self.current_scope.define(name, var_type)

    def lookup(self, name: str):
        return self.current_scope.resolve(name)

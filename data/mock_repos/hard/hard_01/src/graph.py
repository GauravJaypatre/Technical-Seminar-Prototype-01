class Node:
    def __init__(self, name: str):
        self.name = name
        self.dependencies = []

class DependencyGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name: str) -> Node:
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def add_dependency(self, from_node: str, to_node: str):
        parent = self.add_node(from_node)
        child = self.add_node(to_node)
        parent.dependencies.append(child)

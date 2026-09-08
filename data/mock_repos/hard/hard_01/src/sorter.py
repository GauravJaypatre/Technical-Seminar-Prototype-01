from src.graph import DependencyGraph

class CycleError(Exception):
    pass

class TopologicalSorter:
    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def sort(self) -> list[str]:
        visited = set()
        result = []

        def dfs(node):
            # BUG: visited node is flagged as cycle even if fully processed in earlier tree branch
            if node in visited:
                raise CycleError(f"Cycle detected at node {node.name}")
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            result.append(node.name)

        for node in self.graph.nodes.values():
            if node not in visited:
                dfs(node)
        return result

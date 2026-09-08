import pytest
from src.graph import DependencyGraph
from src.sorter import TopologicalSorter, CycleError

def test_diamond_dependency_dag():
    g = DependencyGraph()
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")

    sorter = TopologicalSorter(g)
    # FAILS ON UNPATCHED CODE
    order = sorter.sort()
    assert order.index("D") < order.index("B")
    assert order.index("D") < order.index("C")
    assert order.index("B") < order.index("A")
    assert order.index("C") < order.index("A")

def test_actual_cycle():
    g = DependencyGraph()
    g.add_dependency("X", "Y")
    g.add_dependency("Y", "X")
    sorter = TopologicalSorter(g)
    with pytest.raises(CycleError):
        sorter.sort()

# Issue: False positive cycle detection across disjoint component paths

**Task ID:** `hard_01`  
**Tier:** `hard`  
**Domain:** `Algorithms / Compilers`  

## Description
The topological sorter across `graph.py` and `sorter.py` uses a single `visited` set without a 3-color DFS distinction (visiting vs visited). When a node is reachable via multiple distinct paths in a valid DAG (diamond dependency), it erroneously raises `CycleError`.

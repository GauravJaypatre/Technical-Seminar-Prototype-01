"""
Multi-SLM Swarm Package:
Orchestrates specialized Small Language Models (Router, DeepSeek-Coder Code Analyzer,
Qwen 2.5 QA Verifier) via LangGraph for automated repository bug-fixing.
"""
from src.swarm.agents import (
    OllamaClient,
    RouterAgent,
    CodeAnalyzerAgent,
    QAVerifierAgent
)
from src.swarm.graph import (
    SwarmState,
    build_swarm_graph
)
def run_swarm_baseline(*args, **kwargs):
    from src.swarm.swarm_runner import run_swarm_baseline as _run
    return _run(*args, **kwargs)

__all__ = [
    "OllamaClient",
    "RouterAgent",
    "CodeAnalyzerAgent",
    "QAVerifierAgent",
    "SwarmState",
    "build_swarm_graph",
    "run_swarm_baseline"
]

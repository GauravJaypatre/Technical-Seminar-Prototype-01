"""
Unit and Integration Tests for Multi-SLM Swarm:
1. RouterAgent deterministic heuristic thresholds
2. QAVerifierAgent structural validation and path hallucination rejection
3. LangGraph StateGraph orchestration and conditional retry loop with mocked SLMs
"""
import pytest
from unittest.mock import MagicMock

from src.baseline_client import GenerationResult
from src.swarm.agents import (
    RouterAgent,
    QAVerifierAgent,
    CodeAnalyzerAgent,
    OllamaClient
)
from src.swarm.graph import build_swarm_graph, SwarmState


# =====================================================================
# 1. Router Agent Tests
# =====================================================================

def test_router_easy_single_file_short_issue():
    router = RouterAgent()
    decision = router.route("easy", ["calculator/math_ops.py"], "Fix addition logic bug.")
    assert decision["strategy"] == "direct"
    assert decision["sub_steps"] == 1
    assert "calculator/math_ops.py" in decision["focus_files"]


def test_router_easy_multi_file_triggers_decomposition():
    router = RouterAgent()
    decision = router.route("easy", ["file_a.py", "file_b.py"], "Small typo in two files.")
    assert decision["strategy"] == "decomposed"
    assert decision["sub_steps"] == 2


def test_router_easy_long_issue_triggers_decomposition():
    router = RouterAgent()
    long_issue = "Detailed trace: " + ("x" * 1600)
    decision = router.route("easy", ["file_a.py"], long_issue)
    assert decision["strategy"] == "decomposed"
    assert decision["sub_steps"] == 2


def test_router_medium_single_file_short_issue():
    router = RouterAgent()
    decision = router.route("medium", ["module/core.py"], "Short issue text under 1500 chars.")
    assert decision["strategy"] == "direct"
    assert decision["sub_steps"] == 1


def test_router_medium_multi_file():
    router = RouterAgent()
    decision = router.route("medium", ["module/a.py", "module/b.py"], "Refactor across modules.")
    assert decision["strategy"] == "decomposed"
    assert decision["sub_steps"] == 2


def test_router_hard_always_decomposed():
    router = RouterAgent()
    # Even single short file must be decomposed on hard tier
    decision = router.route("hard", ["algo/graph.py"], "Bug in dijkstra.")
    assert decision["strategy"] == "decomposed"
    assert decision["sub_steps"] == 2


# =====================================================================
# 2. QA Verifier Agent Tests
# =====================================================================

def test_qa_verifier_empty_diff():
    mock_client = MagicMock(spec=OllamaClient)
    verifier = QAVerifierAgent(client=mock_client)
    res = verifier.verify_patch("", ["calculator/ops.py"], "Issue")
    assert res["is_valid"] is False
    assert "empty" in res["critique"].lower()


def test_qa_verifier_missing_headers():
    mock_client = MagicMock(spec=OllamaClient)
    verifier = QAVerifierAgent(client=mock_client)
    bad_diff = """@@ -1,3 +1,3 @@
- old
+ new
"""
    res = verifier.verify_patch(bad_diff, ["calculator/ops.py"], "Issue")
    assert res["is_valid"] is False
    assert "missing unified diff headers" in res["critique"].lower()


def test_qa_verifier_hallucinated_file_path():
    mock_client = MagicMock(spec=OllamaClient)
    verifier = QAVerifierAgent(client=mock_client)
    hallucinated_diff = """--- a/fake/nonexistent/module.py
+++ b/fake/nonexistent/module.py
@@ -1,2 +1,2 @@
- x = 1
+ x = 2
"""
    res = verifier.verify_patch(hallucinated_diff, ["calculator/ops.py"], "Issue")
    assert res["is_valid"] is False
    assert "hallucinated" in res["critique"].lower()


def test_qa_verifier_valid_diff_calls_slm():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.return_value = GenerationResult(
        text='{"is_valid": true, "critique": "Looks good."}',
        prompt_tokens=100,
        completion_tokens=20,
        latency_sec=0.2,
        provider="ollama_local",
        model_name="qwen2.5:0.5b",
        raw_response={}
    )
    verifier = QAVerifierAgent(client=mock_client)
    valid_diff = """--- a/calculator/ops.py
+++ b/calculator/ops.py
@@ -10,3 +10,3 @@
- return a - b
+ return a + b
"""
    res = verifier.verify_patch(valid_diff, ["calculator/ops.py"], "Issue")
    assert res["is_valid"] is True
    assert res["tokens"] == 120


# =====================================================================
# 3. LangGraph Workflow & Retry Routing Tests
# =====================================================================

def test_swarm_graph_clean_single_pass():
    """Valid diff on first try -> terminates with retry_count=0, qa_exhausted=False."""
    mock_client = MagicMock(spec=OllamaClient)
    
    # Analyzer generates valid diff
    mock_client.generate.side_effect = [
        # Analyzer call
        GenerationResult(
            text="```diff\n--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n- a\n+ b\n```",
            prompt_tokens=200,
            completion_tokens=50,
            latency_sec=0.5,
            provider="ollama_local",
            model_name="deepseek-coder:6.7b",
            raw_response={}
        ),
        # QA Verifier call
        GenerationResult(
            text='{"is_valid": true, "critique": "Valid patch."}',
            prompt_tokens=100,
            completion_tokens=15,
            latency_sec=0.1,
            provider="ollama_local",
            model_name="qwen2.5:0.5b",
            raw_response={}
        )
    ]

    app = build_swarm_graph(client=mock_client)

    initial_state: SwarmState = {
        "task_id": "test_01",
        "tier": "easy",
        "task_dir": ".",
        "issue_text": "Simple bug",
        "target_files": ["main.py"],
        "seed": 42,
        "temperature": 0.2,
        "routing_decision": {},
        "candidate_patch": "",
        "previous_patch": None,
        "qa_critique": None,
        "retry_count": 0,
        "is_valid": False,
        "qa_exhausted": False,
        "final_patch": "",
        "agents_involved": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_latency_sec": 0.0
    }

    final_state = app.invoke(initial_state)
    assert final_state["is_valid"] is True
    assert final_state["retry_count"] == 0
    assert final_state["qa_exhausted"] is False
    assert "--- a/main.py" in final_state["final_patch"]
    assert "router" in final_state["agents_involved"]
    assert "code_analyzer" in final_state["agents_involved"]
    assert "qa_verifier" in final_state["agents_involved"]


def test_swarm_graph_retry_recovery():
    """Attempt 1 produces malformed diff (no headers); Attempt 2 produces valid diff."""
    mock_client = MagicMock(spec=OllamaClient)

    mock_client.generate.side_effect = [
        # Analyzer 1: malformed diff
        GenerationResult(
            text="```diff\nJust some text without headers\n```",
            prompt_tokens=150,
            completion_tokens=20,
            latency_sec=0.3,
            provider="ollama_local",
            model_name="deepseek-coder:6.7b",
            raw_response={}
        ),
        # (QA verifier rejects it structurally, no SLM call needed for bad headers)
        # Analyzer 2: valid diff
        GenerationResult(
            text="```diff\n--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n- a\n+ b\n```",
            prompt_tokens=250,
            completion_tokens=60,
            latency_sec=0.6,
            provider="ollama_local",
            model_name="deepseek-coder:6.7b",
            raw_response={}
        ),
        # QA Verifier 2: passes
        GenerationResult(
            text='{"is_valid": true, "critique": "Fixed."}',
            prompt_tokens=120,
            completion_tokens=15,
            latency_sec=0.15,
            provider="ollama_local",
            model_name="qwen2.5:0.5b",
            raw_response={}
        )
    ]

    app = build_swarm_graph(client=mock_client)

    initial_state: SwarmState = {
        "task_id": "test_02",
        "tier": "medium",
        "task_dir": ".",
        "issue_text": "Medium bug",
        "target_files": ["main.py"],
        "seed": 42,
        "temperature": 0.2,
        "routing_decision": {},
        "candidate_patch": "",
        "previous_patch": None,
        "qa_critique": None,
        "retry_count": 0,
        "is_valid": False,
        "qa_exhausted": False,
        "final_patch": "",
        "agents_involved": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_latency_sec": 0.0
    }

    final_state = app.invoke(initial_state)
    assert final_state["is_valid"] is True
    assert final_state["retry_count"] == 1
    assert final_state["qa_exhausted"] is False
    assert "--- a/main.py" in final_state["final_patch"]


def test_swarm_graph_retry_exhaustion():
    """Attempt 1 and Attempt 2 both fail -> loop terminates at retry_count=2 with qa_exhausted=True."""
    mock_client = MagicMock(spec=OllamaClient)

    # Both analyzer attempts generate bad headers
    mock_client.generate.side_effect = [
        # Analyzer 1
        GenerationResult(
            text="```diff\nBad patch attempt 1\n```",
            prompt_tokens=150,
            completion_tokens=20,
            latency_sec=0.3,
            provider="ollama_local",
            model_name="deepseek-coder:6.7b",
            raw_response={}
        ),
        # Analyzer 2
        GenerationResult(
            text="```diff\nBad patch attempt 2\n```",
            prompt_tokens=200,
            completion_tokens=25,
            latency_sec=0.4,
            provider="ollama_local",
            model_name="deepseek-coder:6.7b",
            raw_response={}
        )
    ]

    app = build_swarm_graph(client=mock_client)

    initial_state: SwarmState = {
        "task_id": "test_03",
        "tier": "hard",
        "task_dir": ".",
        "issue_text": "Hard bug",
        "target_files": ["main.py"],
        "seed": 42,
        "temperature": 0.2,
        "routing_decision": {},
        "candidate_patch": "",
        "previous_patch": None,
        "qa_critique": None,
        "retry_count": 0,
        "is_valid": False,
        "qa_exhausted": False,
        "final_patch": "",
        "agents_involved": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_latency_sec": 0.0
    }

    final_state = app.invoke(initial_state)
    assert final_state["is_valid"] is False
    assert final_state["retry_count"] == 2
    assert final_state["qa_exhausted"] is True
    # The last candidate patch is forced through for sandbox grading
    assert "Bad patch attempt 2" in final_state["final_patch"]

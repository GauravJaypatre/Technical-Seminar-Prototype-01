"""
LangGraph Swarm Workflow Module:
Orchestrates the Multi-SLM Swarm using a LangGraph StateGraph:
START -> router_node -> analyzer_node -> verifier_node -> (conditional retry loop <= 2) -> END
"""
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from src.swarm.agents import (
    OllamaClient,
    RouterAgent,
    CodeAnalyzerAgent,
    QAVerifierAgent
)


class SwarmState(TypedDict):
    task_id: str
    tier: str
    task_dir: str
    issue_text: str
    target_files: List[str]
    seed: int
    temperature: float
    routing_decision: Dict[str, Any]
    candidate_patch: str
    previous_patch: Optional[str]
    qa_critique: Optional[str]
    retry_count: int
    is_valid: bool
    qa_exhausted: bool
    final_patch: str
    agents_involved: List[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_latency_sec: float


def build_swarm_graph(
    client: Optional[OllamaClient] = None,
    analyzer_model: str = "deepseek-coder:6.7b",
    verifier_model: str = "qwen2.5:0.5b"
):
    """
    Constructs and compiles the LangGraph StateGraph workflow for the Swarm.
    """
    ollama_client = client or OllamaClient()
    router = RouterAgent()
    analyzer = CodeAnalyzerAgent(client=ollama_client, model_name=analyzer_model)
    verifier = QAVerifierAgent(client=ollama_client, model_name=verifier_model)

    def router_node(state: SwarmState) -> Dict[str, Any]:
        decision = router.route(
            tier=state.get("tier", "medium"),
            target_files=state.get("target_files", []),
            issue_text=state.get("issue_text", "")
        )
        agents = state.get("agents_involved", [])
        if "router" not in agents:
            agents = agents + ["router"]
        return {
            "routing_decision": decision,
            "agents_involved": agents
        }

    def analyzer_node(state: SwarmState) -> Dict[str, Any]:
        patch, p_tok, c_tok, lat = analyzer.generate_patch(
            task_dir=state["task_dir"],
            issue_text=state["issue_text"],
            target_files=state["target_files"],
            routing_decision=state.get("routing_decision", {}),
            qa_critique=state.get("qa_critique"),
            previous_patch=state.get("candidate_patch"),
            seed=state.get("seed", 42),
            temperature=state.get("temperature", 0.2)
        )
        agents = state.get("agents_involved", [])
        if "code_analyzer" not in agents:
            agents = agents + ["code_analyzer"]

        return {
            "candidate_patch": patch,
            "previous_patch": state.get("candidate_patch"),
            "total_prompt_tokens": state.get("total_prompt_tokens", 0) + p_tok,
            "total_completion_tokens": state.get("total_completion_tokens", 0) + c_tok,
            "total_latency_sec": state.get("total_latency_sec", 0.0) + lat,
            "agents_involved": agents
        }

    def verifier_node(state: SwarmState) -> Dict[str, Any]:
        res = verifier.verify_patch(
            candidate_patch=state.get("candidate_patch", ""),
            target_files=state.get("target_files", []),
            issue_text=state.get("issue_text", ""),
            seed=state.get("seed", 42)
        )
        agents = state.get("agents_involved", [])
        if "qa_verifier" not in agents:
            agents = agents + ["qa_verifier"]

        tokens_spent = res.get("tokens", 0)
        lat = res.get("latency", 0.0)
        is_valid = res.get("is_valid", True)
        critique = res.get("critique")

        current_retries = state.get("retry_count", 0)

        if is_valid:
            return {
                "is_valid": True,
                "qa_critique": None,
                "qa_exhausted": False,
                "final_patch": state.get("candidate_patch", ""),
                "total_prompt_tokens": state.get("total_prompt_tokens", 0) + tokens_spent,
                "total_latency_sec": state.get("total_latency_sec", 0.0) + lat,
                "agents_involved": agents
            }
        else:
            new_retry = current_retries + 1
            exhausted = new_retry >= 2
            return {
                "is_valid": False,
                "qa_critique": critique,
                "retry_count": new_retry,
                "qa_exhausted": exhausted,
                # If retries are exhausted, force candidate patch through for sandbox grading
                "final_patch": state.get("candidate_patch", "") if exhausted else "",
                "total_prompt_tokens": state.get("total_prompt_tokens", 0) + tokens_spent,
                "total_latency_sec": state.get("total_latency_sec", 0.0) + lat,
                "agents_involved": agents
            }

    def should_retry(state: SwarmState) -> str:
        # If valid or retries exhausted (>= 2), terminate workflow and hand to sandbox
        if state.get("is_valid", False) or state.get("retry_count", 0) >= 2:
            return "end"
        return "retry"

    # Assemble LangGraph workflow
    workflow = StateGraph(SwarmState)
    workflow.add_node("router", router_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("verifier", verifier_node)

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "analyzer")
    workflow.add_edge("analyzer", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        should_retry,
        {
            "retry": "analyzer",
            "end": END
        }
    )

    return workflow.compile()
